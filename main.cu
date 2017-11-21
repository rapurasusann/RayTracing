
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <GL/glut.h>
#include "camera.h"
#include "sphere.h"


#define HEIGHT 400
#define WIDTH 800
#define SPHERES 3
#define SIZE HEIGHT*WIDTH
#define DEEP 32

__constant__ sphere dev_list[SPHERES];

__device__ vec3 DrawBack(ray r) {
	vec3 unit_direction = unit_vector(r.direction());// 単位方向=単位ベクトル
	float t = 0.5*(unit_direction.y() + 1.0);
	return (1.0 - t)*vec3(1.0, 1.0, 1.0) + t*vec3(0.3, 0.7, 1.0);//背景の色の設定
}

__device__ vec3 SetMetal(ray r, hit_record rec, sphere *list, int depth) {
	for (int i = 0; i < SPHERES; i++) {
		if (list[i].hit(r, 0.001, FLT_MAX, rec))//あたれば処理
		{
			return vec3(rec.color.r(), rec.color.g(), rec.color.b()) / depth;
		}

	}
	return DrawBack(r);
}

__device__ bool Check_Shadow(ray &r, sphere *list, hit_record &temp_rec) {
	hit_record rec;

	for (int i = 0; i < SPHERES; i++) {

		if (list[i].hit(r, 0.001, FLT_MAX, rec))//あたれば処理
		{
			switch (rec.material) {
			case lambertian:
				r = ray(rec.p, rec.normal + random_in_unit_sphere());
				return true;

			case metal:
				r = ray(rec.p, reflect(unit_vector(r.direction()), rec.normal));
				return true;
			}
		}
	}
	temp_rec = rec;
	return false;
}

__device__ vec3 color(ray&r, sphere *list, int depth, hit_record rec)//色を設定する関数
{
	for (int i = 0; i < SPHERES; i++) {

		if (list[i].hit(r, 0.001, FLT_MAX, rec))//あたれば処理
		{
			switch (rec.material) {
			case lambertian:
				return vec3(rec.color.r(), rec.color.g(), rec.color.b()) / float(depth);
				break;

			case metal:
				return vec3(rec.color.r(), rec.color.g(), rec.color.b())*SetMetal(ray(rec.p, reflect(unit_vector(r.direction()), rec.normal)), rec, list, depth);
				break;
			}
		}
	}
}

__global__ void kernel(unsigned char *ptr) {
	__shared__ vec3 col[DEEP];

	float x = ((float)blockIdx.x + XorFrand(0.0, 1.0)) / (float)WIDTH;
	float y = ((float)blockIdx.y + XorFrand(0.0, 1.0)) / (float)HEIGHT;

	int offset = blockIdx.x + blockIdx.y * gridDim.x;
	int cacheIndex = threadIdx.x;

	vec3 lookfrom(0,0, -7);
	vec3 lookat(0, 0, 0);
	float dist_to_foucus = (lookfrom - lookat).length();
	float aperture = 0.0;
	camera cam(lookfrom, lookat, vec3(0, 1, 0), 20, float(WIDTH) / float(HEIGHT), aperture, dist_to_foucus);//カメラクラス camの作成

	bool shadow = false;
	int depth = 0;
	hit_record rec;
	hit_record temp_rec;

	ray r = cam.get_ray(x, y);//カメラクラスに問題あり
	ray temp_r = r;

	for (int i = 0; i < 15; i++) {
		shadow = Check_Shadow(temp_r, dev_list, temp_rec);
		if (shadow == false) {
			break;
		}
		depth++;
	}

	if (depth == 0) {//当たらなければ背景を描画
		col[cacheIndex] = DrawBack(r);
	}

	else if (shadow == false) {
		col[cacheIndex] = color(r, dev_list, depth, temp_rec);//vec3型変数colに、関数colorで出した値を入れる
	}
	else {
		col[cacheIndex] = vec3(1.0, 0.0, 0.0);
	}

	int i = blockDim.x / 2;

	while (i != 0) {
		if (cacheIndex < i) {
			col[cacheIndex] += col[cacheIndex + i];
		}
		i /= 2;
	}

	if (cacheIndex == 0) {
		col[cacheIndex] /= (float)DEEP;
		col[cacheIndex] = vec3(sqrt(col[cacheIndex].r()), sqrt(col[cacheIndex].g()), sqrt(col[cacheIndex].b()));
		ptr[offset * 3 + 0] = int(255.99*col[cacheIndex].r());
		ptr[offset * 3 + 1] = int(255.99*col[cacheIndex].g());
		ptr[offset * 3 + 2] = int(255.99*col[cacheIndex].b());
	}
}

void display() {
	unsigned char *pixels;
	pixels = new unsigned char[WIDTH*HEIGHT * 3];

	unsigned char *dev_pixels;
	cudaMalloc((void**)&dev_pixels, WIDTH*HEIGHT * 3);
	sphere *list;
	list = new sphere[SPHERES];
	list[0] = sphere(vec3(0, 0, -1), 0.5, vec3(0.1, 0.1, 0.8), lambertian);
	list[1] = sphere(vec3(1, 0, -1), 0.5, vec3(1.0, 1.0, 0.3), metal);
	list[2] = sphere(vec3(0, -100.5, -1), 100, vec3(0.3, 0.8, 0.3), lambertian);
	
	cudaMemcpyToSymbol(dev_list, list, sizeof(sphere)*SPHERES);
	
	dim3 grids(WIDTH, HEIGHT);
	kernel << <grids, DEEP >> > (dev_pixels);
	
	cudaMemcpy(pixels, dev_pixels, SIZE * 3, cudaMemcpyDeviceToHost);

	glDrawPixels(WIDTH, HEIGHT, GL_RGB, GL_UNSIGNED_BYTE, pixels);
	glFlush();

	cudaFree(dev_pixels);
	delete[] pixels;
}


int main(int argc, char *argv[])
{
	glutInitWindowPosition(100, 100);
	glutInitWindowSize(WIDTH, HEIGHT);
	glutInit(&argc, argv);
	glutCreateWindow(argv[0]);
	glutDisplayFunc(display);
	glutMainLoop();
	return 0;
}

