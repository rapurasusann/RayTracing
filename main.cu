#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <GL/glut.h>
#include "camera.h"
#include "sphere.h"


#define HEIGHT 400            
#define WIDTH 800
#define SPHERES 3
#define SIZE (HEIGHT*WIDTH)
#define DEEP 100
#define THREAD 16

unsigned char *pixels;        //pixelの色を格納する配列へのポインタ(ホスト用)
unsigned char *dev_pixels;    //pixelの色を格納する配列へのポインタ(デバイス用)
float *temp_pixels;           //pixelの色を格納する配列へのポインタ(平均化用)

sphere *list;                 //球の情報を格納する配列へのポインタ

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

__global__ void kernel(float *ptr) {
	vec3 col;

	int x = threadIdx.x + blockIdx.x*blockDim.x;
	int y = threadIdx.y + blockIdx.y*blockDim.y;

	float w = ((float)x + XorFrand(0.0, 1.0)) / (float)WIDTH;
	float h = ((float)y + XorFrand(0.0, 1.0)) / (float)HEIGHT;

	int offset = x + y *blockDim.x* gridDim.x;
	vec3 lookfrom(0, 0, -7);
	vec3 lookat(0, 0, 0);
	float dist_to_foucus = (lookfrom - lookat).length();
	float aperture = 0.0;
	camera cam(lookfrom, lookat, vec3(0, 1, 0), 20, float(WIDTH) / float(HEIGHT), aperture, dist_to_foucus);//カメラクラス camの作成

	bool shadow = false;
	int depth = 0;
	hit_record rec;
	hit_record temp_rec;

	ray r = cam.get_ray(w, h);
	ray temp_r = r;

	for (int i = 0; i < 15; i++) {
		shadow = Check_Shadow(temp_r, dev_list, temp_rec);
		if (shadow == false) {
			break;
		}
		depth++;
	}

	if (depth == 0) {//当たらなければ背景を描画
		col = DrawBack(r);
	}

	else if (shadow == false) {
		col = color(r, dev_list, depth, temp_rec);//vec3型変数colに、関数colorで出した値を入れる
	}
	else {
		col = vec3(1.0, 0.0, 0.0);
	}

	col = vec3(sqrt(col.r()), sqrt(col.g()), sqrt(col.b()));

	ptr[offset * 3 + 0] += col.r() / DEEP;
	ptr[offset * 3 + 1] += col.g() / DEEP;
	ptr[offset * 3 + 2] += col.b() / DEEP;
}

__global__ void conversion(float *a, unsigned char *b) {
	int index = blockIdx.x;
	b[index * 3 + 0] = (int)(255.99 *a[index * 3 + 0]);
	b[index * 3 + 1] = (int)(255.99 *a[index * 3 + 1]);
	b[index * 3 + 2] = (int)(255.99 *a[index * 3 + 2]);
}

void Initialize() {
	pixels = new unsigned char[SIZE * 3];

	cudaMalloc((void**)&dev_pixels, SIZE * 3);
	cudaMalloc((void**)&temp_pixels, (sizeof(float))*(SIZE * 3));

	list = new sphere[SPHERES];
	list[0] = sphere(vec3(0, 0, -1), 0.5, vec3(0.1, 0.1, 0.8), lambertian);
	list[1] = sphere(vec3(1, 0, -1), 0.5, vec3(1.0, 1.0, 0.3), metal);
	list[2] = sphere(vec3(0, -100.5, -1), 100, vec3(0.3, 0.8, 0.3), lambertian);

	cudaMemcpyToSymbol(dev_list, list, sizeof(sphere)*SPHERES);

	dim3 grids(WIDTH/THREAD, HEIGHT/THREAD);
	dim3 threads(THREAD, THREAD);

	for (int i = 0; i < DEEP; i++) {
		kernel << <grids, threads >> > (temp_pixels);
	}
	conversion << <SIZE, 1 >> > (temp_pixels, dev_pixels);

	cudaMemcpy(pixels, dev_pixels, SIZE * 3, cudaMemcpyDeviceToHost);

}

void display() {
	glDrawPixels(WIDTH, HEIGHT, GL_RGB, GL_UNSIGNED_BYTE, pixels);
	glFlush();
}

void atexit() {
	cudaFree(dev_pixels);
	delete[] pixels;
}

int main(int argc, char *argv[])
{
	glutInitWindowPosition(100, 100);
	glutInitWindowSize(WIDTH, HEIGHT);
	glutInit(&argc, argv);
	glutCreateWindow(argv[0]);
	Initialize();
	glutDisplayFunc(display);
	glutMainLoop();
	atexit();
	return 0;
}
