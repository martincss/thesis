import glob
from absorber_file_generator import generate_one_to_map
from multiprocessing import Pool, cpu_count
from line_sampler_to_mw import sphere_uniform_grid, sample_one_to_map

if cpu_count() == 12: number_of_cores = 5 else number_of_cores = 2
pool = Pool(number_of_cores)

rays_directory = './rays_test/'
absorbers_directory = './rays_test/'

def generate_absorbers_sample(rays_directory, absorbers_directory, pool):

    tasks =[(i,handle, rays_directory, absorbers_directory) for i, handle in \
            enumerate(glob.glob(rays_directory + 'ray*'))]

    pool.map(generate_one_to_map, tasks)



def make_ray_sample_uniform(r_interval, number_of_sightlines, pool):

    theta_interval, phi_interval = sphere_uniform_grid(number_of_sightlines)

    for r in r_interval:

        tasks = [(i, number_of_sightlines, r, theta, phi) for i, (theta, phi) in \
                 enumerate(zip(theta_interval, phi_interval))]
        pool.map(sample_one_to_map, tasks)


if __name__ == '__main__':
    # generate_absorbers_sample(rays_directory, absorbers_directory, pool)

    make_ray_sample_uniform([2000], 10, pool)
