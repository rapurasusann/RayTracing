#ifndef CAMERAH
#define CAMERAH
#include "ray.h"
#define _USE_MATH_DEFINES
#include <math.h>//‰~Žü—¦—p

__device__ static unsigned long xors_x = 123456789;
__device__ static unsigned long xors_y = 362436069;
__device__ static unsigned long xors_z = 521288629;
__device__ static unsigned long xors_w = 88675123;

__device__
unsigned long Xorshift128()
{
	unsigned long t;
	t = (xors_x ^ (xors_x << 11));
	xors_x = xors_y; xors_y = xors_z; xors_z = xors_w;
	return (xors_w = (xors_w ^ (xors_w >> 19)) ^ (t ^ (t >> 8)));
}
__device__
long Xorshift128(long l, long h)
{
	unsigned long t;
	t = (xors_x ^ (xors_x << 11));
	xors_x = xors_y; xors_y = xors_z; xors_z = xors_w;
	xors_w = (xors_w ^ (xors_w >> 19)) ^ (t ^ (t >> 8));
	return l + (xors_w % (h - l));
}

__device__
float XorFrand(float l, float h)
{
	return l + (h - l)*(Xorshift128(0, 100000) / 100000.0f);
}

__device__ vec3 random_in_unit_disk() {
	vec3 p;
	do {
		p = 2.0*vec3(XorFrand(0.0,1.0), XorFrand(0.0, 1.0), XorFrand(0.0, 1.0)) - vec3(1, 1, 1);
	} while (dot(p, p) >= 1.0);
	return p;
}

class camera {
public:
	__device__ __host__ camera(vec3 lookfrom, vec3 lookat, vec3 vup, float vfov, float aspect, float aperture, float focus_dist) { // vfov is top to bottom in degrees
		lens_radius = aperture / 2;
		float theta = vfov*M_PI / 180;
		float half_height = tan(theta / 2);
		float half_width = aspect * half_height;
		origin = lookfrom;
		w = unit_vector(lookfrom - lookat);
		u = unit_vector(cross(vup, w));
		v = cross(w, u);
		lower_left_corner = origin - half_width*focus_dist*u - half_height*focus_dist*v - focus_dist*w;
		horizontal = 2 * half_width*focus_dist*u;
		vertical = 2 * half_height*focus_dist*v;
	}
	__device__ ray get_ray(float s, float t) {
		vec3 rd = lens_radius*random_in_unit_disk();
		vec3 offset = u * rd.x() + v * rd.y();
		return ray(origin + offset, lower_left_corner + s*horizontal + t*vertical - origin - offset);
	}

	vec3 origin;
	vec3 lower_left_corner;
	vec3 horizontal;
	vec3 vertical;
	vec3 u, v, w;
	float lens_radius;
};
#endif



