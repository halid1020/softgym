#pragma once
#include <iostream>
#include <vector>

inline void swap(int &a, int &b) {int tmp =a ; a = b; b=tmp;}

class SoftgymCloth : public Scene
{
public:
    float cam_x;
    float cam_y;
    float cam_z;
    float cam_angle_x;
    float cam_angle_y;
    float cam_angle_z;
    int cam_width;
    int cam_height;
    Colour front_colour;
    Colour back_colour;


	SoftgymCloth(const char* name) : Scene(name) {}

    float get_param_float(py::array_t<float> scene_params, int idx)
    {
        auto ptr = (float *) scene_params.request().ptr;
        float out = ptr[idx];
        return out;
    }

    //params ordering: xpos, ypos, zpos, xsize, zsize, stretch, bend, shear
    // render_type, cam_X, cam_y, cam_z, angle_x, angle_y, angle_z, width, height
	void Initialize(py::array_t<float> scene_params, int thread_idx=0)
    {  
        auto ptr = (float *) scene_params.request().ptr;
	    float initX = ptr[0];
	    float initY = ptr[1];
	    float initZ = ptr[2];

		int dimx = (int)ptr[3];
		int dimz = (int)ptr[4];
		float radius = 0.00625f;

        int render_type = ptr[8]; // 0: only points, 1: only mesh, 2: points + mesh

        cam_x = ptr[9];
        cam_y = ptr[10];
        cam_z = ptr[11];
        cam_angle_x = ptr[12];
        cam_angle_y = ptr[13];
        cam_angle_z = ptr[14];
        cam_width = int(ptr[15]);
        cam_height = int(ptr[16]);

        
        front_colour = Colour(ptr[19], ptr[20], ptr[21]);
        back_colour = Colour(ptr[22], ptr[23], ptr[24]);

        // Cloth
        float stretchStiffness = ptr[5]; //0.9f;
		float bendStiffness = ptr[6]; //1.0f;
		float shearStiffness = ptr[7]; //0.9f;
		int phase = NvFlexMakePhase(0, eNvFlexPhaseSelfCollide | eNvFlexPhaseSelfCollideFilter);
		float mass = float(ptr[17])/(dimx*dimz);	// avg bath towel is 500-700g
        int flip_mesh = int(ptr[18]); // Flip half
	    CreateSpringGrid(Vec3(initX, -initY, initZ), dimx, dimz, 1, radius, phase, stretchStiffness, bendStiffness, shearStiffness, 0.0f, 1.0f/mass);
	    // Flip the last half of the mesh for the folding task
	    if (flip_mesh)
	    {
	        int size = g_buffers->triangles.size();
//	        for (int j=int((dimz-1)*3/8); j<int((dimz-1)*5/8); ++j)
//	            for (int i=int((dimx-1)*1/8); i<int((dimx-1)*3/8); ++i)
//	            {
//	                int idx = j *(dimx-1) + i;
//
//	                if ((i!=int((dimx-1)*3/8-1)) && (j!=int((dimz-1)*3/8)))
//	                    swap(g_buffers->triangles[idx* 3 * 2], g_buffers->triangles[idx*3*2+1]);
//	                if ((i != int((dimx-1)*1/8)) && (j!=int((dimz-1)*5/8)-1))
//	                    swap(g_buffers->triangles[idx* 3 * 2 +3], g_buffers->triangles[idx*3*2+4]);
//                }
	        for (int j=0; j<int((dimz-1)); ++j)
	            for (int i=int((dimx-1)*1/8); i<int((dimx-1)*1/8)+5; ++i)
	            {
	                int idx = j *(dimx-1) + i;

	                if ((i!=int((dimx-1)*1/8+4)))
	                    swap(g_buffers->triangles[idx* 3 * 2], g_buffers->triangles[idx*3*2+1]);
	                if ((i != int((dimx-1)*1/8)))
	                    swap(g_buffers->triangles[idx* 3 * 2 +3], g_buffers->triangles[idx*3*2+4]);
                }
        }
        g_colors[3] = front_colour;
        g_colors[4] = back_colour;
        // for (int i=0; i<int(g_mesh->GetNumVertices()*0.5); ++i)
        //         //g_mesh->m_colours[i] = 1.5f*colors[5];
        //      g_mesh->m_colours[i] = 1.5f*front_colour;

        // for (int i=int(g_mesh->GetNumVertices()*0.5); i<g_mesh->GetNumVertices(); ++i)
        //     //g_mesh->m_colours[i] = 1.5f*colors[6];
        //     g_mesh->m_colours[i] = 1.5f*back_colour;

		g_numSubsteps = 4;
		g_params.numIterations = 30;

		g_params.dynamicFriction = 0.75f;
		g_params.particleFriction = 1.0f;
		g_params.damping = 1.0f;
		g_params.sleepThreshold = 0.02f;

		g_params.relaxationFactor = 1.0f;
		g_params.shapeCollisionMargin = 0.04f;

		g_sceneLower = Vec3(-1.0f);
		g_sceneUpper = Vec3(1.0f);
		g_drawPoints = false;

        g_params.radius = radius*1.8f;
        g_params.collisionDistance = 0.005f;

        g_drawPoints = render_type & 1;
        g_drawCloth = (render_type & 2) >>1;
        g_drawSprings = false;

    }

    virtual void CenterCamera(void)
    {
        g_camPos = Vec3(cam_x, cam_y, cam_z);
        g_camAngle = Vec3(cam_angle_x, cam_angle_y, cam_angle_z);
        g_screenHeight = cam_height;
        g_screenWidth = cam_width;
    }
};