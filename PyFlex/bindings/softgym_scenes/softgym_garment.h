// From https://github.com/Xingyu-Lin/softgym/commit/77722f213044a3d7c10b7497e96f7500784ce33b

#pragma once
#include <iostream>
#include <vector>
#include <cmath>
#include <list>
#include <iterator>
#include <string>



class SoftgymGarment : public Scene
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
    char tshirt_path[100];
    Colour front_colour;
    Colour back_colour;
    Colour inside_colour;
    uint8_t garment_id;
    uint8_t shape_id;

    // maps from vertex index to particle index
    map<uint32_t, uint32_t> vid2pid;
    map<uint32_t, uint32_t> pid2vid;

    // Map from garment id to garment name with strings
    map<uint8_t, string> garment_id2name;

    SoftgymGarment(const char* name): Scene(name) {
        

        // Initialize the garment id to name map
        garment_id2name[0] = "Tshirt";
        garment_id2name[1] = "Trousers";
        garment_id2name[2] = "Dress";
        garment_id2name[3] = "Top";
        garment_id2name[4] = "Jumpsuit";
    }

    std::string make_path(const std::string& path) {
        const char* pyflexroot = std::getenv("PYFLEXROOT");
        if (!pyflexroot) {
            throw std::runtime_error("PYFLEXROOT environment variable not set");
        }

        std::string full_path = pyflexroot;
        full_path += path;

        return full_path;
    }

    float get_param_float(py::array_t<float> scene_params, int idx)
    {
        auto ptr = (float *) scene_params.request().ptr;
        float out = ptr[idx];
        return out;
    }

    void sortInd(uint32_t* a, uint32_t* b, uint32_t* c)
    {
        if (*b < *a)
            swap(a,b);

        if (*c < *b)
        {
            swap(b,c);
            if (*b < *a)
                swap(b, a);
        }
    }


    void findUnique(map<uint32_t, uint32_t> &unique, Mesh* m)
    {
        map<vector<float>, uint32_t> vertex;
        map<vector<float>, uint32_t>::iterator it;

        uint32_t count = 0;
        for (uint32_t i=0; i < m->GetNumVertices(); ++i)
        {
            Point3& v = m->m_positions[i];
            float arr[] = {v.x, v.y, v.z};
            vector<float> p(arr, arr + sizeof(arr)/sizeof(arr[0]));

            it = vertex.find(p);
            if (it == vertex.end()) {
                vertex[p] = i;
                unique[i] = i;
                count++;
            }
            else
            {
                unique[i] = it->second;
            }
        }

        cout << "total vert:  " << m->GetNumVertices() << endl;
        cout << "unique vert: " << count << endl;
    }


    void createTshirt(std::string filename, Vec3 lower, float scale, float rotation, 
        Vec3 velocity, int phase, float invMass, float stiffness, 
        Colour front_colour, Colour back_colour, Colour inside_colour)
    {
        // import the mesh
        //cout << filename << endl;
        // convert filename to char*
        const char* filename_c = filename.c_str();
        Mesh* m = ImportMesh(filename_c);
        //cout << "imported mesh" << endl;
        if (!m)
            return;

        // Get the middle of the meshm then translate to origin
        Vec3 meshLower, meshUpper;
        m->GetBounds(meshLower, meshUpper);

        Vec3 mid = (meshUpper+meshLower)/2;
        Vec3 edges = meshUpper-meshLower;
        float maxEdge = max(max(edges.x, edges.y), edges.z);

        // put mesh at the origin and scale to specified size
        Matrix44 xform = ScaleMatrix(scale/maxEdge)*TranslationMatrix(Point3(-mid));
        m->Transform(xform);
        m->GetBounds(meshLower, meshUpper);


        // Set colors to the g_mesh
        int t1(0), t2(0);
        for (uint32_t i=0; i < m->GetNumVertices(); ++i) {

            if (m->m_positions[i].y > 0.0) {
                m->m_colours[i] = front_colour;
                t1++;
            } else {
                m->m_colours[i] = back_colour;
                t2++;
            }
        }
        
        g_colors[3] = inside_colour;



        // index of particles
        uint32_t baseIndex = uint32_t(g_buffers->positions.size());
        uint32_t currentIndex = baseIndex;

        // find unique vertices
        map<vector<float>, uint32_t> vertex;
        map<vector<float>, uint32_t>::iterator it;

        
        

        // to check for duplicate connections
        map<uint32_t,list<uint32_t> > edgeMap;




        // loop through the faces
        for (uint32_t i=0; i < m->GetNumFaces(); ++i)
        {
            // create particles
            uint32_t a = m->m_indices[i*3+0];
            uint32_t b = m->m_indices[i*3+1];
            uint32_t c = m->m_indices[i*3+2];

            Point3& v0 = m->m_positions[a];
            Point3& v1 = m->m_positions[b];
            Point3& v2 = m->m_positions[c];

            float arr0[] = {v0.x, v0.y, v0.z};
            float arr1[] = {v1.x, v1.y, v1.z};
            float arr2[] = {v2.x, v2.y, v2.z};
            vector<float> pos0(arr0, arr0 + sizeof(arr0)/sizeof(arr0[0]));
            vector<float> pos1(arr1, arr1 + sizeof(arr1)/sizeof(arr1[0]));
            vector<float> pos2(arr2, arr2 + sizeof(arr2)/sizeof(arr2[0]));

            it = vertex.find(pos0);
            if (it == vertex.end()) {
                vertex[pos0] = currentIndex;
                vid2pid[a] = currentIndex++;
                pid2vid[currentIndex] = a;

                Vec3 p0 = lower + meshLower + Vec3(v0.x, v0.y, v0.z);
                g_buffers->positions.push_back(Vec4(p0.x, p0.y, p0.z, invMass));
                g_buffers->velocities.push_back(velocity);
                g_buffers->phases.push_back(phase);
                g_buffers->uvs.push_back({
                    m->m_colours[a].r,
                    m->m_colours[a].g,
                    m->m_colours[a].b});
            }
            else
            {
                vid2pid[a] = it->second;
                pid2vid[it->second] = a;
            }

            it = vertex.find(pos1);
            if (it == vertex.end()) {
                vertex[pos1] = currentIndex;
                vid2pid[b] = currentIndex++;
                pid2vid[currentIndex] = b;

                Vec3 p1 = lower + meshLower + Vec3(v1.x, v1.y, v1.z);
                g_buffers->positions.push_back(Vec4(p1.x, p1.y, p1.z, invMass));
                g_buffers->velocities.push_back(velocity);
                g_buffers->phases.push_back(phase);
                g_buffers->uvs.push_back({
                    m->m_colours[b].r,
                    m->m_colours[b].g,
                    m->m_colours[b].b});
            }
            else
            {
                vid2pid[b] = it->second;
                pid2vid[it->second] = b;
            }

            it = vertex.find(pos2);
            if (it == vertex.end()) {
                vertex[pos2] = currentIndex;
                vid2pid[c] = currentIndex++;
                pid2vid[currentIndex] = c;

                Vec3 p2 = lower + meshLower + Vec3(v2.x, v2.y, v2.z);
                g_buffers->positions.push_back(Vec4(p2.x, p2.y, p2.z, invMass));
                g_buffers->velocities.push_back(velocity);
                g_buffers->phases.push_back(phase);
                g_buffers->uvs.push_back({
                    m->m_colours[c].r,
                    m->m_colours[c].g,
                    m->m_colours[c].b});
            }
            else
            {
                vid2pid[c] = it->second;
                pid2vid[it->second] = c;
            }

            // create triangles
            g_buffers->triangles.push_back(vid2pid[a]);
            g_buffers->triangles.push_back(vid2pid[b]);
            g_buffers->triangles.push_back(vid2pid[c]);

            // connect springs

            // add spring if not duplicate
            list<uint32_t>::iterator it;
            // for a-b
            if (edgeMap.find(a) == edgeMap.end())
            {
                CreateSpring(vid2pid[a], vid2pid[b], stiffness);
                edgeMap[a].push_back(b);
            }
            else
            {

                it = find(edgeMap[a].begin(), edgeMap[a].end(), b);
                if (it == edgeMap[a].end())
                {
                    CreateSpring(vid2pid[a], vid2pid[b], stiffness);
                    edgeMap[a].push_back(b);
                }
            }

            // for a-c
            if (edgeMap.find(a) == edgeMap.end())
            {
                CreateSpring(vid2pid[a], vid2pid[c], stiffness);
                edgeMap[a].push_back(c);
            }
            else
            {

                it = find(edgeMap[a].begin(), edgeMap[a].end(), c);
                if (it == edgeMap[a].end())
                {
                    CreateSpring(vid2pid[a], vid2pid[c], stiffness);
                    edgeMap[a].push_back(c);
                }
            }

            // for b-c
            if (edgeMap.find(b) == edgeMap.end())
            {
                CreateSpring(vid2pid[b], vid2pid[c], stiffness);
                edgeMap[b].push_back(c);
            }
            else
            {

                it = find(edgeMap[b].begin(), edgeMap[b].end(), c);
                if (it == edgeMap[b].end())
                {
                    CreateSpring(vid2pid[b], vid2pid[c], stiffness);
                    edgeMap[b].push_back(c);
                }
            }

        }


        delete m;
    }


    //params ordering: xpos, ypos, zpos, xsize, zsize, stretch, bend, shear
    // render_type, cam_X, cam_y, cam_z, angle_x, angle_y, angle_z, width, height
    void Initialize(py::array_t<float> scene_params, int thread_idx=0)
    {
        cout << "initialize garment" << endl;
        auto ptr = (float *) scene_params.request().ptr;
        
        float initX = ptr[0];
        float initY = ptr[1];
        float initZ = ptr[2];
        float scale = ptr[3];
        float rot = ptr[4];
        float velX = ptr[5];
        float velY = ptr[6];
        float velZ = ptr[7];
        
        float stretchStiffness = ptr[8];
		float bendStiffness = ptr[9];
		float shearStiffness = ptr[10];

        // float stiff = ptr[11];
        float mass = ptr[11];
        float radius = ptr[12];
        cam_x = ptr[13];
        cam_y = ptr[14];
        cam_z = ptr[15];
        cam_angle_x = ptr[16];
        cam_angle_y = ptr[17];
        cam_angle_z = ptr[18];
        cam_width = int(ptr[19]);
        cam_height = int(ptr[20]);
        int render_type = ptr[21]; // 0: only points, 1: only mesh, 2: points + mesh

        front_colour = Colour(ptr[22], ptr[23], ptr[24]);
        back_colour = Colour(ptr[25], ptr[26], ptr[27]);
        inside_colour = Colour(ptr[28], ptr[29], ptr[30]);

        std::string garment_type = garment_id2name[ptr[31]];
        int shape_id = uint(ptr[32]);

        //cout << "finish laoding all params" << endl;



        int phase = NvFlexMakePhase(0, eNvFlexPhaseSelfCollide | eNvFlexPhaseSelfCollideFilter);
        // float static_friction = 0.5;
        // float dynamic_friction = 1.0;


        // create path for the garment from the garment id and shape id, /data/TriGarments/{garment name}/{garment name}_{shape id}.obj
        // make {gament anem} and {shape id} as parameters
        std::string shape_id_padded = std::to_string(shape_id);
        shape_id_padded = std::string(4 - shape_id_padded.length(), '0') + shape_id_padded;
        std::string garment_path = "/data/TriGarments/" + garment_type + "/" + garment_type + "_" + shape_id_padded + ".obj";
        
        // cout << "before create Tshirt" << endl;
        // cout << "tshirt path: " << tshirt_path << endl;
        // cout << "garment path: " << garment_path << endl;
        std::string path = make_path(garment_path);
        // cout << "load garment path: " << path << endl;

        createTshirt(
            path, 
            Vec3(initX, initY, initZ), 
            scale, rot, Vec3(velX, velY, velZ), phase, 1/mass, 
            stretchStiffness,
            front_colour, back_colour, inside_colour);

        // cout << "created Tshirt" << endl;


        g_numSubsteps = 4;
        g_params.numIterations = 30;
        g_params.dynamicFriction = 1.2f; //0.75f;
        g_params.particleFriction = 1.0f;
        g_params.damping = 1.0f;
        g_params.sleepThreshold = 0.02f;
        g_params.relaxationFactor = 1.0f;
        g_params.shapeCollisionMargin = 0.04f;
        g_sceneLower = Vec3(-1.0f);
        g_sceneUpper = Vec3(1.0f);

        g_params.radius = radius*1.8f;
        g_params.collisionDistance =  0.005f;

        g_drawPoints = (render_type & 2) >>1; 
        g_drawCloth = false; //(render_type & 2) >>1;
        g_drawMesh = render_type & 1;
        g_drawSprings = false;
        g_drawDiffuse = false;

        cout << "tris: " << g_buffers->triangles.size() << endl;
        
    }

    virtual void CenterCamera(void)
    {
        g_camPos = Vec3(cam_x, cam_y, cam_z);
        g_camAngle = Vec3(cam_angle_x, cam_angle_y, cam_angle_z);
        g_screenHeight = cam_height;
        g_screenWidth = cam_width;
    }

    // // Update partciels positions to g_mesh
    // void UpdateMesh()
    // {
    //     //cout << "update mesh hello" << endl;
    //     // update mesh by iterating through all mesh vertices
    //     for (int i = 0; i < g_mesh->m_positions.size(); i++)
    //     {
    //         // get particle position
    //         Vec4 pos = g_buffers->positions[vid2pid[i]];

    //         // update mesh position
    //         g_mesh->m_positions[i] = Point3(pos.x, pos.y, pos.z);
    //     }
    //     g_mesh->CalculateNormals();
    // }

    // void Update(py::array_t<float> update_params) {
    //     UpdateMesh();
    // }
};