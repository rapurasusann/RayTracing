#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <GL/glut.h>
#include "camera.h"
#include "sphere.h"
#include <random>

#define HEIGHT 400                                 //ウィンドウの横幅
#define WIDTH 800                                  //ウィンドウの縦幅
#define SIZE (HEIGHT*WIDTH)                        //画面サイズ(pixel数)

#define SPHERES 5                                  //球体の数

#define X_MAX 10                                   //球体を作る範囲xの最大値
#define X_MIN -5                                   //　  　　　　　　 最小値
#define Z_MAX 4                                    //             zの最大値
#define Z_MIN -5                                   //                最小値


#define THREAD 16                                  //並列処理のスレッドの数(一辺)
dim3 grids(WIDTH / THREAD, HEIGHT / THREAD);       //並列処理のブロック数
dim3 threads(THREAD, THREAD);                      //並列処理のスレッド数


//---------------------------------------------------------
//フレームレート表示用
int GLframe = 0;                                    //フレーム数
int GLtimenow = 0;                                  //経過時間
int GLtimebase = 0;                                 //計測開始時間
//---------------------------------------------------------

unsigned char *pixels;                              //pixelの色を格納する配列へのポインタ(ホスト用)
unsigned char *dev_pixels;                          //pixelの色を格納する配列へのポインタ(デバイス用)
float *temp_pixels;                                 //pixelの色を格納する配列へのポインタ(平均化用)

float *dev_random;                                  //ランダムな値を格納する配列へのポインタ

int sub_pixels = 0;                                 //SSAAのサブピクセル数

sphere *list;                                       //球の情報を格納する配列へのポインタ

//------------------------------------------------------------------------
//カメラクラスの初期値
vec3 lookfrom(12, 3, 12);                           //視点
vec3 lookat(0, 0, 0);                               //注視店
float dist_to_foucus = (lookfrom - lookat).length();//視点から注視店までの距離
float aperture = 0.0;                               //カメラの絞り ピントの調整
camera cam(lookfrom, lookat, vec3(0, 1, 0), 20, float(WIDTH) / float(HEIGHT), aperture, dist_to_foucus);//カメラクラス camの作成
//-------------------------------------------------------------------------

__constant__ sphere dev_list[SPHERES];              //球体のデータを格納するコンスタントメモリ

//-------------------------------------------------------------------------
//デバイス関数
__device__ vec3 random_in_unit_sphere(float *random, int offset) {
	vec3 p;
	float i = 0.0;
	do {
		p = 2.0*vec3(XorFrand(0.0, random[offset * 3 + 0] + i), XorFrand(0.0, random[offset * 3 + 1] + i), XorFrand(0.0, random[offset * 3 + 2] + i)) - vec3(1.0, 1.0, 1.0);
		i += 0.02;
	} while (p.squared_length() >= 1.0);
	return p;
}
__device__ vec3 DrawBack(ray r) {
	vec3 unit_direction = unit_vector(r.direction());// 単位方向=単位ベクトル
	float t = 0.5*(unit_direction.y() + 1.0);
	return (1.0 - t)*vec3(1.0, 1.0, 1.0) + t*vec3(0.3, 0.7, 1.0);//背景の色の設定
}
__device__ bool hit(const ray& r, float t_min, float t_max, hit_record&rec) {
	hit_record temp_rec;
	bool hit_anything = false;
	double closest_so_far = t_max;
	for (int i = 0; i < SPHERES; i++) {
		if (dev_list[i].hit(r, t_min, closest_so_far, temp_rec)) {//球体すべての当たり判定をチェックする
			hit_anything = true;
			closest_so_far = temp_rec.t;
			rec = temp_rec;
			rec.i = i;
		}
	}
	return hit_anything;
}
__device__ vec3 Setmetal(ray r, hit_record rec, int depth,float *random,int offset) {
	for (int j = 0; j < depth; j++) {
		if (hit(r, 0.001, FLT_MAX, rec)) {
			{
				switch (dev_list[rec.i].material) {
				case lambertian:
					return vec3(rec.color.r(), rec.color.g(), rec.color.b()) / depth;

				case metal:
					r = ray(rec.p, reflect(unit_vector(r.direction() + dev_list[rec.i].fuzz * random_in_unit_sphere(random,offset)), rec.normal));
					if (j == depth - 1) {
						return vec3(rec.color.r(), rec.color.g(), rec.color.b()) / depth;
					}
					break;

				case dielectric:
					vec3 reflected = reflect(r.direction(), rec.normal);
					vec3 refracted;
					vec3 outward_normal;
					float ni_over_nt;
					float reflect_prob;
					float cosine;

					if (dot(r.direction(), rec.normal) > 0) {
						outward_normal = -rec.normal;
						ni_over_nt = dev_list[rec.i].ref_idx;
						cosine = dev_list[rec.i].ref_idx * dot(r.direction(), rec.normal) / r.direction().length();
					}
					else {
						outward_normal = rec.normal;
						ni_over_nt = 1.0 / dev_list[rec.i].ref_idx;
						cosine = -dot(r.direction(), rec.normal) / r.direction().length();
					}
					if (reflact(r.direction(), outward_normal, ni_over_nt, refracted)) {
						reflect_prob = schlick(cosine, dev_list[rec.i].ref_idx);
					}
					else {
						reflect_prob = 1.0;
					}
					if (XorFrand(0.0, random[offset]) < reflect_prob) {
						r = ray(rec.p, reflected);
					}
					else {
						r = ray(rec.p, refracted);
					}
					break;
				}
			}
		}
	}
	return DrawBack(r);
}
__device__ vec3 Setdie(ray r, hit_record rec, int depth,float *random, int offset) {
	for (int j = 0; j < depth; j++) {
		if (hit(r, 0.001, FLT_MAX, rec)) {
			{
				switch (dev_list[rec.i].material) {
				case lambertian:
					return vec3(rec.color.r(), rec.color.g(), rec.color.b())/depth;

				case metal:
					r = ray(rec.p, reflect(unit_vector(r.direction() + dev_list[rec.i].fuzz * random_in_unit_sphere(random,offset)), rec.normal));
					if (j == depth - 1) {
						return vec3(rec.color.r(), rec.color.g(), rec.color.b());
					}
					break;

				case dielectric:
					vec3 reflected = reflect(r.direction(), rec.normal);
					vec3 refracted;
					vec3 outward_normal;
					float ni_over_nt;
					float reflect_prob;
					float cosine;

					if (dot(r.direction(), rec.normal) > 0) {
						outward_normal = -rec.normal;
						ni_over_nt = dev_list[rec.i].ref_idx;
						cosine = dev_list[rec.i].ref_idx * dot(r.direction(), rec.normal) / r.direction().length();
					}
					else {
						outward_normal = rec.normal;
						ni_over_nt = 1.0 / dev_list[rec.i].ref_idx;
						cosine = -dot(r.direction(), rec.normal) / r.direction().length();
					}
					if (reflact(r.direction(), outward_normal, ni_over_nt, refracted)) {
						reflect_prob = schlick(cosine, dev_list[rec.i].ref_idx);
					}
					else {
						reflect_prob = 1.0;
					}
					if (XorFrand(0.0, random[offset]) < reflect_prob) {
						r = ray(rec.p, reflected);
					}
					else {
					r = ray(rec.p, refracted);
					}
					break;
				}
			}
		}
	}
	return DrawBack(r);
}
__device__ bool Check_Shadow(ray &r, float *random, int offset) {
	hit_record rec;
	if (hit(r, 0.01, FLT_MAX, rec)) {
		switch (rec.material) {
		case lambertian:
			vec3 target = rec.p + rec.normal + random_in_unit_sphere(random, offset);
			r = ray(rec.p, target - rec.p);
			return true;

		case metal:
			r = ray(rec.p, reflect(unit_vector(r.direction() + dev_list[rec.i].fuzz * random_in_unit_sphere(random, offset)), rec.normal));
			return true;

		case dielectric:
			vec3 reflected = reflect(r.direction(), rec.normal);
			vec3 refracted;
			vec3 outward_normal;
			float ni_over_nt;
			float reflect_prob;
			float cosine;

			if (dot(r.direction(), rec.normal) > 0) {
				outward_normal = -rec.normal;
				ni_over_nt = dev_list[rec.i].ref_idx;
				cosine = dev_list[rec.i].ref_idx * dot(r.direction(), rec.normal) / r.direction().length();
			}
			else {
				outward_normal = rec.normal;
				ni_over_nt = 1.0 / dev_list[rec.i].ref_idx;
				cosine = -dot(r.direction(), rec.normal) / r.direction().length();
			}
			if (reflact(r.direction(), outward_normal, ni_over_nt, refracted)) {
				reflect_prob = schlick(cosine, dev_list[rec.i].ref_idx);
			}
			else {
				reflect_prob = 1.0;
			}
			if (XorFrand(0.0, random[offset]) < reflect_prob) {
				r = ray(rec.p, reflected);
			}
			else {
				r = ray(rec.p, refracted);
			}
			return true;
		}
	}
	return false;
}
__device__ vec3 color(ray& r, int depth, float *random, int offset) {
	hit_record rec;
	if (hit(r, 0.01, FLT_MAX, rec)) {
		switch (rec.material) {
		case lambertian:
			return rec.color / depth;

		case metal:
			return rec.color * Setmetal(ray(rec.p, reflect(unit_vector(r.direction() + dev_list[rec.i].fuzz*random_in_unit_sphere(random, offset)), rec.normal)), rec, depth, random, offset);

		case dielectric:
			vec3 outward_normal;
			vec3 reflected = reflect(r.direction(), rec.normal);
			float ni_over_nt;
			vec3 refracted;
			float reflect_prob;
			float cosine;

			if (dot(r.direction(), rec.normal) > 0) {
				outward_normal = -rec.normal;
				ni_over_nt = dev_list[rec.i].ref_idx;
				cosine = dev_list[rec.i].ref_idx * dot(r.direction(), rec.normal) / r.direction().length();
			}
			else {
				outward_normal = rec.normal;
				ni_over_nt = 1.0 / dev_list[rec.i].ref_idx;
				cosine = -dot(r.direction(), rec.normal) / r.direction().length();
			}
			if (reflact(r.direction(), outward_normal, ni_over_nt, refracted)) {
				reflect_prob = schlick(cosine, dev_list[rec.i].ref_idx);
			}
			else {
				reflect_prob = 1.0;
			}
			if (random[offset]< reflect_prob) {
				r = ray(rec.p, reflected);
		    }
			else {
				r = ray(rec.p, refracted);
			}
			return vec3(1.0, 1.0, 1.0)* Setdie(r, rec, depth, random, offset);
		}
	}
	return DrawBack(r);
}
//--------------------------------------------------------------------------


//--------------------------------------------------------------------------
//グローバル関数
__global__ void kernel(float *ptr, int sub_pixels, float *random, camera cam) {
	vec3 col;

	int x = threadIdx.x + blockIdx.x*blockDim.x;
	int y = threadIdx.y + blockIdx.y*blockDim.y;

	int offset = x + y *blockDim.x* gridDim.x;

	float w = ((float)x + XorFrand(0.0, random[offset])) / (float)WIDTH;
	float h = ((float)y + XorFrand(0.0, random[offset])) / (float)HEIGHT;

	int depth = 0;
	hit_record rec;
	hit_record temp_rec;
	bool dielectric = false;

	ray r = cam.get_ray(w, h);//カメラクラスに問題あり

	ray temp_r = r;

	for (int i = 0; i < 15; i++) {
		if (!Check_Shadow(temp_r, random, offset)) {
			break;
		}
		depth++;
	}

	if (depth == 10) {
		col = vec3(1.0, 0.0, 0.0);
	}
	else {
		col = color(r, depth,random,offset);
	}

	col = vec3(sqrt(col.r()), sqrt(col.g()), sqrt(col.b()));

	ptr[offset * 3 + 0] = (ptr[offset * 3 + 0] / sub_pixels)*(sub_pixels - 1) + col.r() / sub_pixels;
	ptr[offset * 3 + 1] = (ptr[offset * 3 + 1] / sub_pixels)*(sub_pixels - 1) + col.g() / sub_pixels;
	ptr[offset * 3 + 2] = (ptr[offset * 3 + 2] / sub_pixels)*(sub_pixels - 1) + col.b() / sub_pixels;
}
__global__ void conversion(float *a, unsigned char *b) {

	int index = (threadIdx.x+blockIdx.x*blockDim.x) + (threadIdx.y+ blockIdx.y*blockDim.y)*blockDim.x*gridDim.x;

	b[index * 3 + 0] = (int)(255.99 *a[index * 3 + 0]);
	b[index * 3 + 1] = (int)(255.99 *a[index * 3 + 1]);
	b[index * 3 + 2] = (int)(255.99 *a[index * 3 + 2]);
}
__global__ void set_random(float *random) {

	int index = (threadIdx.x + blockIdx.x*blockDim.x) + (threadIdx.y + blockIdx.y*blockDim.y)*blockDim.x*gridDim.x;

	random[index * 3 + 0] = XorFrand(0.0, (float)(threadIdx.x * threadIdx.y) / (float)(blockDim.x*blockDim.y));
	random[index * 3 + 1] = XorFrand(0.0, (float)(threadIdx.x * threadIdx.y) / (float)(blockDim.x*blockDim.y));
	random[index * 3 + 2] = XorFrand(0.0, (float)(threadIdx.x * threadIdx.y) / (float)(blockDim.x*blockDim.y));

	printf("%d:%f\n", blockIdx.x*blockIdx.y, XorFrand(0.0,1.0));
}
//--------------------------------------------------------------------------

float drand48() {
	return rand() % 100 / float(100);
}

void random_scene() {
	int n = 500;
	sphere *list = new sphere[n + 1];
	list[0] = sphere(vec3(0, -10000, 0), 10000, vec3(0.5, 0.5, 0.5), lambertian);//巨大な地面用の球体を作成
	int count = 1;
	for (int x = X_MIN; x < X_MAX; x++) {
		for (int z = Z_MIN; z < Z_MAX; z++) {
			if (count == SPHERES - 4) {                             //球体が上限だったら作成をやめる
				x = X_MAX;
				break;
			}
			vec3 center(x * 1.3+drand48(), 0.2, z * 1.3+drand48());//球体の中心地の設定

			//素材の抽選
			int choose_mat = rand() % 2;
			switch (choose_mat) {

				//通常の素材
			case lambertian:
				list[count++] = sphere(center, 0.2, vec3(drand48()*drand48(), drand48()*drand48(), drand48()*drand48()), lambertian);
				break;

				//金属
			case metal:
				list[count++] = sphere(center, 0.2, vec3(0.5*(1 + drand48()), 0.5*(1 + drand48()), 0.5*(1 + drand48())), metal, 0.5*drand48());
				break;

				//ガラス
			case dielectric:
				//list[count++] = sphere(center, 0.2, dielectric, 1.5);
				break;
			}
		}
	}
	list[count++] = sphere(vec3(0, 1, 0), 1.0, vec3(0.8, 0.8, 0.0), metal, 0.0);
	//list[count++] = sphere(vec3(0, 1, 0), -0.99, dielectric, 1.5);
	list[count++] = sphere(vec3(-4, 1, 0), 1.0, vec3(0.4, 0.2, 0.1), lambertian);
	list[count++] = sphere(vec3(4, 1, 0), 1.0, vec3(0.7, 0.6, 0.5), metal, 0.0);

	cudaMemcpyToSymbol(dev_list, list, sizeof(sphere)*count);//球体のデータをデバイス側にコピーする
}

void SetColor() {
	set_random << <grids, threads >> > (dev_random);
	kernel << <grids, threads >> > (temp_pixels, sub_pixels, dev_random, cam);
	conversion << <grids, threads >> > (temp_pixels, dev_pixels);
	cudaMemcpy(pixels, dev_pixels, SIZE * 3, cudaMemcpyDeviceToHost);
}

void idle() {
	sub_pixels++;//サブピクセルを増やす(画面の鮮明度UP)
	SetColor();
	glutPostRedisplay();

	//GLframe++; //フレーム数を＋１
	//GLtimenow = glutGet(GLUT_ELAPSED_TIME);//経過時間を取得

	//if (GLtimenow - GLtimebase > 1000)	//１秒以上たったらfpsを出力
	//{
	//	printf("fps:%f\r", GLframe*1000.0 / (GLtimenow - GLtimebase));
	//	GLtimebase = GLtimenow;//基準時間を設定		
	//	GLframe = 0;//フレーム数をリセット
	//}
}

void display() {
	glDrawPixels(WIDTH, HEIGHT, GL_RGB, GL_UNSIGNED_BYTE, pixels);
	glFlush();
	//printf("%d回目の更新\n", sub_pixels);
}

void Initialize() {
	pixels = new unsigned char[SIZE * 3];
	cudaMalloc((void**)&dev_pixels, SIZE * 3);
	cudaMalloc((void**)&temp_pixels, (sizeof(float))*(SIZE * 3));
	cudaMalloc((void**)&dev_random, (sizeof(float))*(SIZE * 3));

	random_scene();
}

void Atexit() {
	cudaFree(dev_pixels);
	cudaFree(temp_pixels);
	cudaFree(dev_random);

	delete[] pixels;
}

int main(int argc, char *argv[])
{
	glutInitWindowPosition(0, 0);
	glutInitWindowSize(WIDTH, HEIGHT);
	glutInit(&argc, argv);
	glutCreateWindow(argv[0]);
	Initialize();              //初期化関数
	glutDisplayFunc(display);  //描画関数
	glutIdleFunc(idle);        //アイドル関数(更新処理)
	glutMainLoop();            //メインループ
	Atexit();                  //終了時に呼ばれる関数
	return 0;
}
