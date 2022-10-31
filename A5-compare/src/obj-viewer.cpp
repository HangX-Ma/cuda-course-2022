#if defined(WIN32)
#define WIN32_LEAN_AND_MEAN
#  include <windows.h>
#endif

#include "kernel.h"
#include <GL/glh_glut.h>
#include <stdio.h>
#include "licence.h"

bool b[256];
int win_w = 512, win_h = 512;

using namespace glh;
glut_simple_mouse_interactor object;
void CaptureScreen(int, int);

float DISP_SCALE = 0.001f;
char *dataPath;
int stFrame = 0;
float p[3], q[3];

float gDepOff = 0.025;

// for sprintf
#pragma warning(disable: 4996)

extern void initModel(const char *);
extern void quitModel();
extern void drawModel(bool, bool, bool, bool, int);
extern void updateModel();
extern void buildBVH();

extern void doIt();
extern void buildIt();

static int level = 1;

float lightpos[4] = {13, 10.2, 3.2, 0};

// check for OpenGL errors
void checkGLError()
{
    GLenum error;
    while ((error = glGetError()) != GL_NO_ERROR) {
		char msg[512];
		sprintf(msg, "error - %s\n", (char *) gluErrorString(error));
		printf(msg);
    }
}

void initSetting()
{
	b['9'] = false;
}

void initOpengl()
{
	glClearColor(1.0, 1.0, 1.0, 1.0);

	// initialize OpenGL lighting
	GLfloat lightPos[] =   {10.0, 10.0, 10.0, 0.0};
	GLfloat lightAmb[4] =  {0.0, 0.0, 0.0, 1.0};
	GLfloat lightDiff[4] = {1.0, 1.0, 1.0, 1.0};
	GLfloat lightSpec[4] = {1.0, 1.0, 1.0, 1.0};

	glLightfv(GL_LIGHT0, GL_POSITION, &lightpos[0]);
	glLightfv(GL_LIGHT0, GL_AMBIENT, lightAmb);
	glLightfv(GL_LIGHT0, GL_DIFFUSE, lightDiff);
	glLightfv(GL_LIGHT0, GL_SPECULAR, lightSpec);

	//glLightModeli(GL_LIGHT_MODEL_COLOR_CONTROL_EXT, GL_SEPARATE_SPECULAR_COLOR_EXT);
	GLfloat black[] =  {0.0, 0.0, 0.0, 1.0};
	glLightModelfv(GL_LIGHT_MODEL_AMBIENT, black);
	glLightModeli(GL_LIGHT_MODEL_TWO_SIDE, GL_TRUE);

	glEnable(GL_LIGHT0);
	glEnable(GL_LIGHTING);
	glEnable(GL_DEPTH_TEST);
}

void updateFPS()
{
}

void begin_window_coords()
{
    glMatrixMode(GL_PROJECTION);
    glPushMatrix();
    glLoadIdentity();
    glOrtho(0.0, win_w, 0.0, win_h, -1.0, 1.0);
    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();
}

void end_window_coords()
{
    glMatrixMode(GL_PROJECTION);
    glPopMatrix();
    glMatrixMode(GL_MODELVIEW);
}

void drawGround()
{
	glEnable(GL_COLOR_MATERIAL);
	glColorMaterial(GL_FRONT_AND_BACK, GL_AMBIENT_AND_DIFFUSE);

	glBegin(GL_QUADS);
	glColor3f(1.f, 0.f, 0.f);
	glVertex3f(20, 0, 20);
	glVertex3f(-20, 0, 20);
	glVertex3f(-20, 0, -20);
	glVertex3f(20, 0, -20);
	glEnd();

	glDisable(GL_COLOR_MATERIAL);
}

extern void drawEdges(bool, bool);
extern void drawVFs(int);
extern void drawDebugVF(int);

void drawOther()
{
	glDisable(GL_LIGHTING);

	glEnable(GL_LIGHTING);
}


void draw()
{
	glPushMatrix();
	glRotatef(-90, 1, 0, 0);

	drawModel(!b['t'], !b['p'], !b['s'], b['e'], level);

	/*	glPushMatrix();
	glTranslatef(0, 0, -0.5f);
	glutSolidSphere(0.3, 20, 20);
	glPopMatrix();
*/
	glPopMatrix();

	if (b['g'])
		drawGround();
}

static bool ret = false;

void display()
{
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    glDisable(GL_DEPTH_TEST);
	glDisable(GL_LIGHTING);
	glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
	glShadeModel(GL_SMOOTH);

	if (!b['B']) {
    // gradient background
    begin_window_coords();
    glBegin(GL_QUADS);
        glColor3f(0.2, 0.4, 0.8);
        glVertex2f(0.0, 0.0);
        glVertex2f(win_w, 0.0);
        glColor3f(0.05, 0.1, 0.2);
        glVertex2f(win_w, win_h);
        glVertex2f(0, win_h);
    glEnd();
    end_window_coords();
	}

    glMatrixMode(GL_MODELVIEW);

	glLoadIdentity();
    object.apply_transform();

	if (b['v']	) {
		GLdouble modelMatrix[16];
		glGetDoublev(GL_MODELVIEW_MATRIX, modelMatrix);
		FILE *fp=fopen("c:\\temp\\view.dat", "wb");
		fwrite(modelMatrix, sizeof(GLdouble), 16, fp);
		fclose(fp);
		b['v'] = false;
	}

	// draw scene
	if (b['w'])
		glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);
	else
		glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);

	glEnable(GL_LIGHTING);

	glEnable(GL_DEPTH_TEST);
	glEnable(GL_NORMALIZE);

	draw();

    glutSwapBuffers();
	updateFPS();
	//checkGLError();

	if(b['x'] && ret)   {
		CaptureScreen(512, 512);
	}
}

void idle_fem()
{
	if (b[' '])
		object.trackball.increment_rotation();

	if (b['d']) {
		doIt();
	}

	glutPostRedisplay();
}

void quit()
{
	quitModel();
	cudaFree(adjInfo);
	cudaFree(gIntensityIn_d_);
	cudaFree(gIntensityOut_d_);
	exit(0);
}

void key_fem(unsigned char k, int x, int y)
{
	b[k] = !b[k];

	switch (k) {
	case '=':
		level++;
		break;

	case '-':
		level--;
		break;

	case 27:
	case 'q':
		quit();
		break;

	case 'b':
		buildIt();
		break;

	case 's':
		doIt();
		break;
	}

	object.keyboard(k, x, y);
	glutPostRedisplay();
}

void resize(int w, int h)
{
    if (h == 0) h = 1;

    glViewport(0, 0, w, h);
    
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    gluPerspective(60.0, (GLfloat)w/(GLfloat)h, 0.1, 500.0);
    
    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();

    object.reshape(w, h);

    win_w = w; win_h = h;
}

void mouse(int button, int state, int x, int y)
{
    object.mouse(button, state, x, y);
}

void motion(int x, int y)
{
    object.motion(x, y);
}

void main_menu(int i)
{
    key_fem((unsigned char) i, 0, 0);
}

void initMenu()
{    
    glutCreateMenu(main_menu);
	glutAddMenuEntry("Toggle animation [d]", 'd');
	glutAddMenuEntry("Toggle obb/aabb [o]", 'o');
	glutAddMenuEntry("========================", '=');
	glutAddMenuEntry("Toggle rebuild/refit  (aabb) [r]", 'r');
	glutAddMenuEntry("Increasing boxes level(aabb) [=]", '=');
	glutAddMenuEntry("Decreasing boxes level(aabb) [-]", '-');
	glutAddMenuEntry("========================", '=');
    glutAddMenuEntry("Toggle wireframe [w]", 'w');
	glutAddMenuEntry("Toggle lighting [l]", 'l');
	glutAddMenuEntry("Toggle avi recording [x]", 'x');
	glutAddMenuEntry("Save camera[s]", 's');
	glutAddMenuEntry("Reset camera[t]", 't');
	glutAddMenuEntry("========================", '=');
    glutAddMenuEntry("Quit/q [esc]", '\033');
    glutAttachMenu(GLUT_RIGHT_BUTTON);
}

void usage_self_cd()
{
	printf("Keys:\n");
	//printf("c - Checking self-collisions.\n");
	printf("t - Toggle display colliding/all triangles.\n");
	printf("q/ESC - Quit.\n\n");
	printf("Mouse:\n");
	printf("Left Btn - Obit.\n");
	printf("Ctrl+Left Btn - Zoom.\n");
	printf("Shift+Left Btn - Pan.\n");
}

//FEM
int main(int argc, char **argv)
{
	if (argc < 2) {
		printf("usage: %s input.obj\n", argv[0]);
		return -1;
	}

	_CHK_LICENCE

		usage_self_cd();

	dataPath = argv[1];

	glutInit(&argc, argv);
	glutInitDisplayMode(GLUT_DOUBLE | GLUT_DEPTH | GLUT_RGBA | GLUT_STENCIL);
	glutInitWindowSize(win_w, win_h);
	glutCreateWindow("Cloth Viewer");

	initOpengl();
	initModel(dataPath);
	buildBVH();

	object.configure_buttons(1);
	object.dolly.dolly[2] = -3;
	object.trackball.incr = rotationf(vec3f(1, 1, 0), 0.05);

	glutDisplayFunc(display);
	glutMouseFunc(mouse);
	glutMotionFunc(motion);
	glutIdleFunc(idle_fem);
	glutKeyboardFunc(key_fem);
	glutReshapeFunc(resize);

	initMenu();

	initSetting();

	glutMainLoop();

	quit();
	return 0;
}

void CaptureScreen(int Width, int Height)
{
#ifdef WIN32
	static int captures=0;
	char filename[20];

	sprintf( filename, "Data/%04d.bmp", captures );
	captures++;

	BITMAPFILEHEADER bf;
	BITMAPINFOHEADER bi;

	char *image = new char[Width*Height*3];
	FILE *file = fopen( filename, "wb");

	if( image!=NULL )
	{
		if( file!=NULL ) 
		{
			glReadPixels( 0, 0, Width, Height, GL_BGR_EXT, GL_UNSIGNED_BYTE, image );

			memset( &bf, 0, sizeof( bf ) );
			memset( &bi, 0, sizeof( bi ) );

			bf.bfType = 'MB';
			bf.bfSize = sizeof(bf)+sizeof(bi)+Width*Height*3;
			bf.bfOffBits = sizeof(bf)+sizeof(bi);
			bi.biSize = sizeof(bi);
			bi.biWidth = Width;
			bi.biHeight = Height;
			bi.biPlanes = 1;
			bi.biBitCount = 24;
			bi.biSizeImage = Width*Height*3;

			fwrite( &bf, sizeof(bf), 1, file );
			fwrite( &bi, sizeof(bi), 1, file );
			fwrite( image, sizeof(unsigned char), Height*Width*3, file );

			fclose( file );
		}
		delete[] image;
	}
#endif
}
