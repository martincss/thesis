from numpy import pi
import glob
from absorber_file_generator import generate_one_to_map
from multiprocessing import Pool, cpu_count
from line_sampler_to_mw import sphere_uniform_grid, sample_one_to_map, \
                               sample_subhalos_one_to_map
from line_sampler_to_any import sample_one_to_map_from_any, \
                                subhalo_fuzzy_sampling_one_to_map
from R_N_fig1_to_M31_and_away import sightlines_filenames, make_figure, \
                                     all_distances
from tools import usable_cores
from tools import get_sun_position_2Mpc_LG as sun


number_of_cores = usable_cores()
pool = Pool(number_of_cores)
#pool = Pool()

rays_directory = './rays_2Mpc_LG_to_mw_2000_wrt_mwcenter_sun/'
#rays_directory = './rays_2Mpc_LG_to_m31_and_away_sun/'
#rays_directory = './rays_test/'
#absorbers_directory = './absorbers_2Mpc_LG_to_mw_2000_wrt_mwcenter_sun/'
absorbers_directory = './absorbers_2Mpc_LG_from_subhalos_fuzzy/'

rays_subhalo_directory = './rays_2Mpc_LG_from_subhalos/'
rays_subhalo_fuzzy_directory = './rays_2Mpc_LG_from_subhalos_fuzzy/'

def generate_absorbers_sample(rays_directory, absorbers_directory, pool):

    tasks =[(i,handle, rays_directory, absorbers_directory) for i, handle in \
            enumerate(glob.glob(rays_directory + 'ray*'))]

    # hardcoding this now because absorber 0.858_6.031 won't generate
    tasks.pop(57)

    pool.map(generate_one_to_map, tasks)



def make_ray_sample_uniform(r_interval, number_of_sightlines, pool, sun=False):

    theta_interval, phi_interval = sphere_uniform_grid(number_of_sightlines)

    for r in r_interval:


        if sun:

            tasks = [(r, theta, phi, sun(), rays_directory) for theta, phi in \
            zip(theta_interval, phi_interval)]
            pool.map(sample_one_to_map_any, tasks)

        else:

            tasks = [(i, number_of_sightlines, r, theta, phi, rays_directory) \
            for i, (theta, phi) in enumerate(zip(theta_interval, phi_interval))]
            pool.map(sample_one_to_map, tasks)


def sample_m31_and_away(r_interval, rays_directory, pool, to_mw = True):
    """
    Samples rays on fixed directions to m31 and away from m31, varying distance
    to endpoints.
    """

    theta_m31 = 2*pi/9
    phi_m31 = 6*(2*pi)/9

    theta_away = 3*pi/9
    phi_away = 2*(2*pi)/9

    thetas = [theta_m31, theta_away]
    phis = [phi_m31, phi_away]

    number_of_sightlines = 2*len(r_interval)//2

    if to_mw:

        tasks = [(i, number_of_sightlines, r, theta, phi, rays_directory) for \
                 i, r in enumerate(r_interval) for (theta, phi) \
                 in zip(thetas, phis)]

        pool.map(sample_one_to_map, tasks)

    # if not going to MW, then rays to sun
    else:

        tasks = [(r, theta, phi, sun(), rays_directory) for r in \
                 r_interval for (theta, phi) in zip(thetas, phis)]

        pool.map(sample_one_to_map_from_any, tasks)


def generate_RN_fig1_distance_frames(distances, pool):

    tasks = [(r, sightlines_filenames(r)) for r in distances]

    pool.map(make_figure, tasks)


def generate_subhalo_samples(number_of_subhalos, ray_end,
                             subhalo_rays_directory, pool):

    tasks = [(subhalo_idx, subhalo_rays_directory, ray_end) for \
             subhalo_idx in range(number_of_subhalos + 1) if subhalo_idx != 1]

    pool.map(sample_subhalos_one_to_map, tasks)



def generate_subhalo_fuzzy_samples(number_of_subhalos, ray_end,
                                  subhalo_rays_directory, number_of_rays, pool):

    tasks = [(subhalo_idx, ray_end, subhalo_rays_directory, number_of_rays) for\
             subhalo_idx in range(number_of_subhalos) if subhalo_idx != 1]
    # sample from subhalos excluding number 1 (i.e. MW)

    pool.map(subhalo_fuzzy_sampling_one_to_map, tasks)




if __name__ == '__main__':
    generate_absorbers_sample(rays_subhalo_fuzzy_directory, absorbers_directory, pool)

    #make_ray_sample_uniform([2000], 500, pool)
    #sample_m31_and_away(all_distances[1:], rays_directory, pool, to_mw=False)
    #generate_RN_fig1_distance_frames(all_distances[1:], pool)

#    generate_subhalo_fuzzy_samples(15, sun(), rays_subhalo_fuzzy_directory,
#                                 10, pool)
    #generate_subhalo_samples(15, sun(), rays_subhalo_directory, pool)
