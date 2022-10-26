#include "obj-viewer.h"
#include "triangle.h"
#include "vector3f.h"
#include <thrust/host_vector.h>

const struct OBJ_COLOR OBJ_COLOR;
struct camera camera;

bool render_mode; // true = solid body, false = wireframe

const float ZOOM_SPEED = 0.1f;
const float ROTATE_SPEED = 0.1f;
float       DISTANCE = 4.0f;

extern std::uint32_t gNumObjects;
extern lbvh::triangle_t* gTriangles;
extern lbvh::vec3f* gVertices;
extern lbvh::vec3f* gNormals;

extern thrust::host_vector<float> gIntensity_h_;
extern volatile bool dstOut;
extern void startHeatTransfer(); 


void switch_render_mode(bool mode) {
    render_mode = mode;
}


void calculate_normal(lbvh::vec3f vecA, lbvh::vec3f vecB, lbvh::vec3f vecC, GLdouble *normal) {
    /* normal x, y, z */
    normal[0] = (vecB.y - vecA.y) * (vecC.z - vecA.z) - (vecC.y - vecA.y) * (vecB.z - vecA.z);
    normal[1] = (vecB.z - vecA.z) * (vecC.x - vecA.x) - (vecB.x - vecA.x) * (vecC.z - vecA.z);
    normal[2] = (vecB.x - vecA.x) * (vecC.y - vecA.y) - (vecC.x - vecA.x) * (vecB.y - vecA.y);
}


void init() {
    glShadeModel(GL_SMOOTH);
    glClearColor(0.0f, 0.0f, 0.0f, 0.0f);
    glClearDepth(1.0f);
    glEnable(GL_DEPTH_TEST);
    glDepthFunc(GL_LEQUAL);
    glEnable(GL_COLOR);
    glEnable(GL_COLOR_MATERIAL);
    glHint(GL_PERSPECTIVE_CORRECTION_HINT, GL_NICEST);

    glEnable(GL_LIGHTING);
    glEnable(GL_NORMALIZE);
    glEnable(GL_LIGHT1);
    GLfloat lightAmbient1[4] = {0.2, 0.2, 0.2, 1.0};
    GLfloat lightPos1[4] = {0.5, 0.5, 0.5, 1.0};
    GLfloat lightDiffuse1[4] = {0.8, 0.8, 0.8, 1.0};
    GLfloat lightSpec1[4] = {1.0, 1.0, 1.0, 1.0};
    GLfloat lightLinAtten = 0.0f;
    GLfloat lightQuadAtten = 1.0f;
    glLightfv(GL_LIGHT1, GL_POSITION, (GLfloat *) &lightPos1);
    glLightfv(GL_LIGHT1, GL_AMBIENT, (GLfloat *) &lightAmbient1);
    glLightfv(GL_LIGHT1, GL_DIFFUSE, (GLfloat *) &lightDiffuse1);
    glLightfv(GL_LIGHT1, GL_SPECULAR, (GLfloat *) &lightSpec1);
    glLightfv(GL_LIGHT1, GL_LINEAR_ATTENUATION, &lightLinAtten);
    glLightfv(GL_LIGHT1, GL_QUADRATIC_ATTENUATION, &lightQuadAtten);
    glLightModeli(GL_LIGHT_MODEL_TWO_SIDE, GL_TRUE);
}

void draw_obj() {
    if (render_mode){
        glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
    }
    else {
        glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);
    }
 
    for (int idx = 0; idx < gNumObjects; idx++) {
        lbvh::vec3f vecA = gVertices[gTriangles[idx].a.vertex_index];
        lbvh::vec3f vecB = gVertices[gTriangles[idx].b.vertex_index];
        lbvh::vec3f vecC = gVertices[gTriangles[idx].c.vertex_index];
        lbvh::vec3f nrmA = gNormals[gTriangles[idx].a.normal_index];
        lbvh::vec3f nrmB = gNormals[gTriangles[idx].b.normal_index];
        lbvh::vec3f nrmC = gNormals[gTriangles[idx].c.normal_index];

        if (dstOut == NULL) {
            glEnable(GL_LIGHTING);
            glBegin(GL_TRIANGLES);
            glColor3f(OBJ_COLOR.red, OBJ_COLOR.green, OBJ_COLOR.blue);
            glNormal3dv((const GLdouble*)nrmA.v);
            glVertex3dv((const GLdouble*)vecA.v);
            glNormal3dv((const GLdouble*)nrmB.v);
            glVertex3dv((const GLdouble*)vecB.v);
            glNormal3dv((const GLdouble*)nrmC.v);
            glVertex3dv((const GLdouble*)vecC.v);
            glEnd();
            glDisable(GL_LIGHTING);
        }
        else {
            float tt = gIntensity_h_[idx];
            GLfloat matDiff[4] = { 1.0, 1.0, 0.0, 1.0 };
            matDiff[1] = 1 - tt;
            matDiff[0] = tt;

            glEnable(GL_LIGHTING);
            glBegin(GL_TRIANGLES);
            glMaterialfv(GL_FRONT, GL_DIFFUSE, matDiff);
            glNormal3dv((const GLdouble*)nrmA.v);
            glVertex3dv((const GLdouble*)vecA.v);
            glNormal3dv((const GLdouble*)nrmB.v);
            glVertex3dv((const GLdouble*)vecB.v);
            glNormal3dv((const GLdouble*)nrmC.v);
            glVertex3dv((const GLdouble*)vecC.v);
            glEnd();
            glDisable(GL_LIGHTING);
        }
    }

    glFlush();
}


void reshape(int w, int h) {
    glViewport(0, 0, w, h);
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    if (h == 0) {
        gluPerspective(80, (float) w, 1.0, 5000.0);
    } else {
        gluPerspective(80, (float) w / (float) h, 1.0, 5000.0);
    }
    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();
}

void arrow_keys(int key, int x, int y) {
    switch (key) {
        case GLUT_KEY_UP: {
            DISTANCE -= ZOOM_SPEED;
            break;
        }
        case GLUT_KEY_DOWN: {
            DISTANCE += ZOOM_SPEED;
            break;
        }
        case GLUT_KEY_LEFT: {
            camera.theta -= ROTATE_SPEED;
            break;
        }
        case GLUT_KEY_RIGHT:
            camera.theta += ROTATE_SPEED;
            break;
        default:
            break;
    }
}


void keyboard(unsigned char key, int x, int y) {
    switch (key) {
        case 27:
            exit(0);
        case 's':
            render_mode = true;
            break;
        case 'w':
            render_mode = false;
            break;
        case 'd':
            startHeatTransfer();
            break;
        default:
            break;
    }
}


void display() {
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    glLoadIdentity();

    camera.x =        DISTANCE * cos(camera.phi)*sin(camera.theta);
    camera.y = 2.0f + DISTANCE * sin(camera.phi)*sin(camera.theta);
    camera.z =        DISTANCE * cos(camera.theta);

    // gluLookAt(camera.x, camera.y, camera.z, 0, 2.0f, 0, 0.0f, 1.0f, 0.0f);
    gluLookAt(camera.x, camera.y, camera.z, 0, 0.0f, 0.0f, 0.0f, 2.0f, 0.0f);
    draw_obj();
    glutSwapBuffers();
    glutPostRedisplay();
}