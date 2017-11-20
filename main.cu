#define HEIGTH 200
#define WIDTH 100
#definr SPHERES 2

__constant__ sphere dev_list[SPHERES]

__device__ color(const ray &r){
    vec3 unit_direction = unitvector(r.direction());
    float t = 0.5*(unit_direction.y() + 1.0);
    return (1.0-t) * vec3(1.0,1.0,1.0) + t*vec3(0.5,0.7,1.0);
}

___global__ kernel(unsigned char &ptr){

    int x = blockidx.x/WIDTH;
    int y = blockidx.y/HEIGTH;
    int offset = blockIdx.x +blockIdx.y * blockDim.x;

    camera cam;
    ray r = cam.get_ray(x,y);

    vec3 col;
    col = color(r);

    ptr[offset + 0] = int(255.99*col.r()) ;
    ptr[offset + 1] = int (255.99*col.g()) ;
    ptr[offset + 2] = int (255.99*col.b());
}

void display(){
    unsigned char *pixels;
    pixels = new unsigned char[WIDTH*HEIGTH*3];
    
    unsigned char *dev_pixels;

    cudaMalloc((void**)&dev_pixels,WIDTH*HEIGTH*3);

    sphere *list;
    list = new spheres[SPHERES];
    list[0] = sphere((0,0,-1),0.5);
    list[1] = sphere(0,-100,-1),100);
    
    cudaMalloc()
    cudaMemcpyToSymbol(dev_list,list,sizeof(spnere*SPHERES));


    dim3 grids(WIDTH,HEIGTH);
    kernel<<<grids,1>>>(dev_pixels);

    cudaMemcpy(pixels,dev_pixels,SIZE, cudaMemcpyDeviceToHost);
    glDrawPixels( WIDTH, HEIGTH, GL_RGBA, GL_UNSIGNED_BYTE, &pixels );

    delete[] pixels;
}
