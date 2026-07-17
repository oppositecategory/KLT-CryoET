import numpy as np 
import scipy

import jax
import jax.numpy as jnp 

from kltpicker_3d.alt_least_squares import alternating_least_squares_solver
from kltpicker_3d.spectral_estimation import estimate_isotropic_powerspectrum_tensor
from kltpicker_3d.fredholm_solver import solve_radial_fredholm_equation
from kltpicker_3d.utils import * 

# Compiled functions 
vect_spectrum_estimation = jax.vmap(estimate_isotropic_powerspectrum_tensor,
                                    in_axes=(0,None))
vect_radial_average = jax.vmap(radial_average_jax,
                               in_axes=(0,None,None,None))

class KLTParticleDetector3D:
    """Detect particles in a tomogram using data-adaptive 3D KLT templates.

    ``particle_diameter`` may be expressed in physical units (for example,
    Angstrom).  ``mgscale`` is the corresponding conversion to voxels per
    input unit.  All geometric calculations after initialization use the
    derived ``*_voxels`` attributes exclusively.
    """

    def __init__(self, 
                 tomogram,
                 particle_diameter: float,
                 mgscale: float,
                 num_particles: int, 
                 legendre_order:int = 150,
                 threshold: float = 0,
                 max_iter: int = 500, 
                 max_order: int = 10,
                 template_size_fraction: float = 0.8):
        if particle_diameter <= 0:
            raise ValueError("particle_diameter must be positive")
        if mgscale <= 0:
            raise ValueError("mgscale must be positive")
        if not 0 < template_size_fraction <= 1:
            raise ValueError("template_size_fraction must be in (0, 1]")

        self.tomogram = tomogram 
        self.particle_diameter = float(particle_diameter)
        self.mgscale = float(mgscale)
        self.max_order = max_order
        self.template_size_fraction = float(template_size_fraction)
        self.bandlimit = np.pi

        scaled_diameter = self.mgscale * self.particle_diameter
        self.particle_diameter_voxels = int(np.floor(scaled_diameter))

        def odd_floor(value):
            size = int(np.floor(value))
            return size if size % 2 else size - 1

        self.patch_size = odd_floor(0.8 * scaled_diameter)
        self.template_diameter = odd_floor(
            self.template_size_fraction * scaled_diameter
        )
        if self.particle_diameter_voxels < 3 or self.patch_size < 3 or self.template_diameter < 3:
            raise ValueError(
                "particle_diameter * mgscale is too small to form a "
                "three-voxel template"
            )

        # The radial Fredholm problem is solved on the same spherical support
        # as the template grid, not in the original physical units.
        self.template_radius_voxels = (self.template_diameter - 1) / 2
        self.max_iter = max_iter 
        
        S = 2 * self.patch_size - 1
        uniform_points, shell_ids, counts = generate_uniform_radial_sampling_points(S, self.bandlimit)
        self.uniform_points = uniform_points
        self.shell_ids = shell_ids
        self.counts = counts

        self.legendre_order = legendre_order
        self.num_particles = num_particles
        self.threshold = threshold

        self.eigvals = None 
        self.eiguncs = None 
        self.score_mat = None
        self.whitened_tomogram = None
        self.initial_particle_psd = None
        self.initial_noise_psd = None
        self.particle_psd = None
        self.noise_psd = None
        self.noise_variance = None
        self.radial_eigvals = None
        self.template_orders = None
        self.template_m_values = None

    def process_tomogram(self):
        # First estimate: calibrate the ALS spectra to spatial variances, then
        # use the calibrated noise spectrum to prewhiten the full tomogram.
        initial_particle_psd, initial_noise_psd, _ = self.factorize_RPSD(
            self.tomogram
        )
        self.initial_particle_psd = initial_particle_psd
        self.initial_noise_psd = initial_noise_psd
        self.whitened_tomogram = prewhiten_tomogram(
            self.tomogram,
            self.uniform_points,
            initial_noise_psd,
        )

        # Second estimate: the templates and likelihood scale must come from
        # the same (whitened) data on which detection is performed.
        particle_psd, noise_psd, noise_var_approx = self.factorize_RPSD(
            self.whitened_tomogram
        )
        self.particle_psd = particle_psd
        self.noise_psd = noise_psd
        self.noise_variance = noise_var_approx
        
        c = self.bandlimit 
        a = self.template_radius_voxels
        K = self.legendre_order
        X,w = scipy.special.roots_legendre(K)
        X_scaled = c/2*X + c/2
        Gx = trigonometric_interpolation(self.uniform_points, 
                                        particle_psd, 
                                        X_scaled)
        # Trigonometric interpolation can overshoot slightly, but a PSD cannot
        # be negative.
        Gx = np.maximum(np.asarray(Gx), 0.0)
        
        eigvals, eigfuncs = [],[]
        max_order = self.max_order
        for i in range(max_order):
            lambdas, funcs,W = solve_radial_fredholm_equation(
                Gx,
                i,
                a,
                c,
                K=K,
            )
            eigvals.append(lambdas)
            eigfuncs.append(funcs)

        eigfuncs = np.array(eigfuncs).reshape(-1,K)
        eigvals = np.array(eigvals).reshape(-1)

        orders = np.tile(np.arange(max_order).reshape(-1, 1), (1, K)).reshape(-1)

        idx = np.argsort(eigvals)[::-1]
        orders = orders[idx]

        eigfuncs = eigfuncs[idx,:]
        eigvals = eigvals[idx]

        idx = np.where(eigvals > np.spacing(1))[0]
        eigvals = eigvals[idx]
        eigfuncs = eigfuncs[idx,:]
        orders = orders[idx]

        templates, eigvals = self.create_GPSF_templates(eigvals,
                                                        eigfuncs,
                                                        orders, 
                                                        Gx)

        num_detected,coords = self.detect_particles(templates, 
                                                    noise_var_approx,
                                                    self.whitened_tomogram)
        return num_detected,coords

    def factorize_RPSD(self, tomogram=None):
        """Estimate and variance-calibrate particle and noise radial PSDs."""
        source = self.tomogram if tomogram is None else tomogram
        M = int(self.patch_size)
        max_d = int(np.floor(0.3*M))

        micro_size = np.min(source.shape)
        m = int(np.floor(micro_size/ M))
        if m < 1:
            raise ValueError(
                "tomogram dimensions must be at least as large as patch_size"
            )

        t = source[:m*M, :m*M, :m*M]
        # (m*M, m*M, m*M) -> (m, M, m, M, m, M) -> (m, m, m, M, M, M)
        patches = t.reshape(m, M, m, M, m, M).transpose(0, 2, 4, 1, 3, 5)
        # Flatten patch index to match (m**3, M, M, M)
        patches = patches.reshape(m**3, M, M, M)

        patches = patches - jnp.mean(patches, axis=(1,2,3)).reshape(-1,1,1,1)

        patches_var = jnp.var(patches,axis=(1,2,3))
        sorted_patches_var = patches_var.sort()
        noise_patch_count = max(
            1,
            int(np.floor(0.25 * patches_var.size)),
        )
        noise_var_approx = jnp.mean(
            sorted_patches_var[:noise_patch_count]
        )
        mean_patch_variance = jnp.mean(patches_var)

        psds = vect_spectrum_estimation(patches,max_d)
        #rblocks = np.array([radial_average(psds[k], bins, len(bins)) for k in range(patches.shape[0])])
        rblocks = vect_radial_average(psds, self.shell_ids, self.counts,self.uniform_points.shape[0])
        factorization = alternating_least_squares_solver(rblocks,self.max_iter,1e-4)
        particle_psd, noise_psd = calibrate_radial_psds(
            self.uniform_points,
            factorization.gamma,
            factorization.v,
            noise_var_approx,
            mean_patch_variance,
        )
        return particle_psd, noise_psd, float(noise_var_approx)

    def create_GPSF_templates(self,
                              eigvals,
                              eigfuncs,
                              orders,
                              G):
        """
            Generates 3D templates in Generalized Prolate Spherodial Function basis
            using radial solutions of KLT equations and their spectrum.

            args:
                orders: the N orders of the corresponding solutions
                G: particle function's radial PSD 
            
            returns:
                templates: flattened complete spherical-harmonic multiplets,
                    shaped (num_modes, template_size, template_size,
                    template_size)
                eigvals: one eigenvalue for every returned angular mode
        """
        a = self.template_radius_voxels
        c = self.bandlimit 
        template_size = self.template_diameter
        K = self.legendre_order 

        radmax = np.floor((template_size-1)/2)
        grid = np.arange(-radmax,radmax+1,1)
        X,Y,Z = np.meshgrid(grid,grid,grid)
        r_tensor = np.sqrt(X**2 + Y**2 + Z**2)
        #r_tensor = np.where(r_tensor > self.bandlimit, 0, r_tensor)
        rho_uniform, idx = np.unique(r_tensor,return_inverse=True)

        # Legendre roots for both integrals
        rho_leg, w = scipy.special.roots_legendre(K)
        rho_leg_a =  (a * 0.5) * rho_leg + a * 0.5 
        rho_leg_c =  (c * 0.5)* rho_leg + c * 0.5
        
        # Each radial eigenvalue of order ell occurs for every
        # m=-ell,...,ell. Account for that (2*ell+1)-fold degeneracy when
        # retaining 99% of the covariance energy.
        truncate_idx = radial_mode_truncation_index(
            eigvals,
            orders,
            energy_fraction=0.99,
        )

        eigfuncs = eigfuncs[:truncate_idx,...]
        eigvals = eigvals[:truncate_idx,...]
        orders = orders[:truncate_idx,...]

        self.eigfuncs = eigfuncs 
        self.radial_eigvals = eigvals

        # We interpolate the radial solutions into uniform radial basis
        # using the Fredholm equation (re-expressing new values of R_{N,m} using  
        # values of it at Legendre roots.
        r_grid_uni = np.outer(rho_uniform, rho_leg_c)
        r_grid_leg = np.outer(rho_leg_a, rho_leg_c)

        def Hn(x,N):
            return 4*np.pi * ((1j**N) * scipy.special.spherical_jn(N,x))
        
        max_N = int(orders.max()) + 1

        # Hn evaluated at multiples of uniform radial points in [0,a]
        Hn_uniform = np.array(
            [Hn(r_grid_uni,N) for N in range(max_N)]
        )

        # Hn evaluated at multiples Legendre roots in [0,a]
        Hn_leg = np.array(
            [Hn(r_grid_leg,N) for N in range(max_N)]
        )

        Hn_leg = Hn_leg[orders]
        Hn_uniform = Hn_uniform[orders]

        sgn = np.where(orders % 2 == 1, -1, 1)
        D = c * 0.5* w * G * (rho_leg_c**2)
        W = a * 0.5 * w* rho_leg_a**2

        H_right = sgn[:,None,None]* Hn_leg
        psi = (Hn_uniform * D[None,None,:]) @ H_right

        # Shape: truncate_idx X rho_uniform.length 
        eigfuncs_uniform = np.einsum('bik,k,bk->bi', psi,W,eigfuncs) / eigvals[:,None]
        radial_templates = eigfuncs_uniform[:,idx]

        templates, radial_indices, m_values = (
            expand_spherical_harmonic_templates(
                radial_templates,
                orders,
                X,
                Y,
                Z,
            )
        )
        template_eigvals = eigvals[radial_indices]

        self.eigvals = template_eigvals
        self.template_orders = orders[radial_indices]
        self.template_m_values = m_values
        return templates, template_eigvals
    

    def detect_particles(self,
                         templates, 
                         noise_var_approx,
                         tomogram=None):
        """ GPU-Accelerated particle detection.
            Function apply FFT-based convolution to run each generated template-kernel across
            the whole 3D tomogram. 
        """
        source = self.tomogram if tomogram is None else tomogram
        n_templates, nx, ny, nz = templates.shape
        psi = templates.reshape(n_templates, nx * ny * nz)
        eigvals_r = jnp.asarray(self.eigvals)
        if eigvals_r.size != n_templates:
            raise ValueError(
                "each spherical-harmonic template must have one eigenvalue"
            )

        Q,R = jnp.linalg.qr(psi.T, mode="reduced")
        H = (
            (R * eigvals_r[None,:]) @ jnp.conj(R.T)
            + noise_var_approx * jnp.eye(R.shape[0])
        )
        H_eigvals,P = jnp.linalg.eigh(H)
        D = (1.0/noise_var_approx) - (1.0/H_eigvals)

        mu = jnp.linalg.slogdet((1/ noise_var_approx) * H)[1]
        D, P = D[::-1],P[:,::-1]
        B = Q @ P
        num_kernels = B.shape[1]
        kernels = B.T.reshape(num_kernels, nx,ny,nz)

        x_num = source.shape[0] - nx + 1
        y_num = source.shape[1] - ny + 1
        z_num = source.shape[2] - nz + 1
        init = jnp.zeros((x_num, y_num, z_num), dtype=jnp.result_type(source, jnp.float32))

        def body(i, acc):
            k = jnp.conj(jnp.flip(kernels[i], axis=(0,1,2)))
            r = jax.scipy.signal.fftconvolve(source, k, mode="valid")
            return acc + D[i] * (jnp.abs(r) ** 2)

        score_mat = jax.lax.fori_loop(0, kernels.shape[0], body, init)
        score_mat = np.array(score_mat - mu)
        self.score_mat = score_mat 
        num_particles, coords = self.picking_from_scoring_vol_3d(score_mat)
        return num_particles, coords
    
    def picking_from_scoring_vol_3d(self, score_vol):
        num_particles = self.num_particles
        max_iter = self.max_iter
        threshold = self.threshold
        offset = self.template_diameter // 2 

        # Distinct non-overlapping particles should have centers separated by
        # approximately one particle diameter. Suppress all competing maxima
        # within that distance to avoid duplicate picks from the same particle.
        r_del = self.particle_diameter_voxels
        log_max = np.max(score_vol)
        eps = 1e-12

        particle_list = []

        num_limit = np.inf if num_particles == -1 else int(num_particles)
        pick_limit = int(min(max_iter, num_limit))
        candidate_indices, candidate_values = ranked_local_maxima_nms_3d(
            score_vol,
            radius=r_del,
            max_picks=pick_limit,
        )

        for (ix, iy, iz), p_max in zip(candidate_indices, candidate_values):
            p_norm = p_max / (log_max + 1e-12)

            if not (p_norm > threshold):
                break

            # Convert valid-score index -> tomogram center coordinate
            cx = ix + offset
            cy = iy + offset
            cz = iz + offset

            particle_list.append([cx, cy, cz, p_max / (log_max + eps)])

        particle_coords = np.array(particle_list, dtype=np.float64) if particle_list else np.zeros((0, 4))
        num_picked_particles = particle_coords.shape[0]

        return num_picked_particles, particle_coords

# def old_prewhiten_tomogram(tomograms, factorization):
#     """ 
#         The function whitens each sub-tomogram using the extracted noise spectrum.

#         Args:
#             tomograms: a tensor containing K sub-tomograms of length N
#             factorization: A dataclass of type RPSDFactorization with ALS solution

#         Returns: 
#             whitened_tomograms: tensor containing K whitend sub-tomograms

#     """
#     _,N,_,_ = tomograms.shape
    
#     uniform_points, _ = generate_uniform_radial_sampling_points(N)
#     noise_rpsd = factorization.v

#     grid = jnp.arange(-(N-1), N) * jnp.pi / N
#     i,j,k = jnp.meshgrid(grid,grid,grid)
#     r_matrix = jnp.sqrt(i**2 + j**2 + k**2)
#     magnitudes, idx = jnp.unique(r_matrix, return_inverse=True)
#     nodes = magnitudes[magnitudes < uniform_points[-1]*jnp.pi]

#     interpolated_noise_rpsd = trigonometric_interpolation(uniform_points*np.pi, noise_rpsd, nodes)
#     noise_rpsd_mat = jnp.pad(interpolated_noise_rpsd, 
#                       (0,
#                        magnitudes.size - interpolated_noise_rpsd.size),
#                       'constant',
#                       constant_values=interpolated_noise_rpsd[-1])
#     noise_rpsd_mat = jnp.reshape(noise_rpsd_mat[idx], [grid.size, grid.size, grid.size])
    
#     whitened_tomograms = vect_prewhite_patch(tomograms,noise_rpsd_mat)
#     whitened_tomograms -= jnp.mean(whitened_tomograms)
#     whitened_tomograms /= jnp.linalg.norm(whitened_tomograms)
#     return whitened_tomograms
