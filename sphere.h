#ifndef SPHEREH
#define SPHEREH

#include "ray.h"

enum material {
	lambertian,
	metal,
};

struct hit_record
{
	float t;
	vec3 p;
	vec3 normal;
	vec3 color;
	material material;
};

__device__ vec3 random_in_unit_sphere() {
	vec3 p;
	do {
		p = 2.0*vec3(XorFrand(0.0, 1.0), XorFrand(0.0, 1.0), XorFrand(0.0, 1.0)) - vec3(1.0, 1.0, 1.0);
	} while (p.squared_length() >= 1.0);
	return p;
}

__device__ vec3 reflect(const vec3& v, const vec3& n) {
	return v - 2 * dot(v, n)*n;
}

__device__ float schlick(float cosine, float ref_idx) {
	float r0 = (1 - ref_idx) / (1 + ref_idx);
	r0 = r0*r0;
	return r0 + (1 - r0)*pow((1 - cosine), 5);
}

__device__ bool refract(const vec3& v, const vec3& n, float ni_over_nt, vec3& refracted) {
	vec3 uv = unit_vector(v);
	float dt = dot(uv, n);
	float discriminant = 1.0 - ni_over_nt*ni_over_nt*(1 - dt*dt);
	if (discriminant > 0) {
		refracted = ni_over_nt*(uv - n*dt) - n*sqrt(discriminant);
		return true;
	}
	else
		return false;
}

class sphere{
public:
	__device__ __host__ sphere() {}
	__device__ __host__ sphere(vec3 cen, float r, vec3 col,material mat) : center(cen), radius(r), color(col),material(mat) {};
	bool hit(const ray& r, float tmin, float tmax, hit_record& rec)const;
	vec3 center;
	float radius;
	vec3 color;
	material material;
};

__device__ bool sphere::hit(const ray& r, float t_min, float t_max, hit_record& rec)const{
	vec3 oc = r.origin() - center;
	float a = dot(r.direction(), r.direction());
	float b = dot(oc, r.direction());
	float c = dot(oc, oc) - radius*radius;
	float discriminant = b*b - a*c;
	if (discriminant > 0) {
		float temp = (-b - sqrt(discriminant)) / a;
		if (temp < t_max && temp > t_min) {
			rec.t = temp;
			rec.p = r.point_at_parameter(rec.t);
			rec.normal = (rec.p - center) / radius;
			rec.color = color;
			rec.material = material;
			return true;
		}
		temp = (-b + sqrt(discriminant)) / a;
		if (temp < t_max && temp > t_min) {
			rec.t = temp;
			rec.p = r.point_at_parameter(rec.t);
			rec.normal = (rec.p - center) / radius;
			rec.color = color;
			rec.material = material;
			return true;
		}
	}
	return false;
}


#endif


