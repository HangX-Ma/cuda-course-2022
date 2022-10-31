#include "cmesh.h"
#include <set>
#include <iostream>

using namespace std;
#include "mat3f.h"
#include "box.h"

#include <stdio.h>

// initModel
#if USE_GPU
#include <cuda_runtime.h>
#include <helper_cuda.h>
#include <helper_functions.h>
#endif

//extern int findCudaDevice(int argc, const char ** argv);

#include <omp.h>

# define	TIMING_BEGIN \
	{double tmp_timing_start = omp_get_wtime();

# define	TIMING_END(message) \
	{double tmp_timing_finish = omp_get_wtime();\
	double  tmp_timing_duration = tmp_timing_finish - tmp_timing_start;\
	printf("%s: %2.5f seconds\n", (message), tmp_timing_duration);}}


//#define POVRAY_EXPORT
#define OBJ_DIR "e:\\temp\\output-objs\\"

//#define VEC_CLOTH

#pragma warning(disable: 4996)

mesh *cloths[16];
mesh *lions[16];

vector<int> vtx_set;
vector<int> dummy_vtx;

set<int> cloth_set;
set<int> lion_set;
set<int> dummy_set;

BOX g_box;
static int sidx=0;


#ifdef HI_RES
int Nx = 501;
int Nz = 501;
double xmin = 0.f, xmax = 500.f;
double zmin = 0.f, zmax = 500.f;
#else
int Nx = 101;
int Nz = 101;
double xmin = 0.f, xmax = 200.f;
double zmin = 0.f, zmax = 200.f;
#endif

#include "cmesh.h"
//#include "mesh_defs.h"
#include <vector>
using namespace std;

// for fopen
#pragma warning(disable: 4996)


bool readobjfile(const char *path, 
				 unsigned int &numVtx, unsigned int &numTri, 
				 tri3f *&tris, vec3f *&vtxs, double scale, vec3f shift, bool swap_xyz, vec2f *&texs, tri3f *&ttris, unsigned int &numTex)
{
	vector<tri3f> triset;
	vector<vec3f> vtxset;
	vector<vec2f> texset;
	vector<tri3f> ttriset;

	FILE *fp = fopen(path, "rt");
	if (fp == NULL) return false;

	char buf[1024];
	while (fgets(buf, 1024, fp)) {
		if (buf[0] == 'v' && buf[1] == ' ') {
				double x, y, z;
				sscanf(buf+2, "%lf%lf%lf", &x, &y, &z);

				if (swap_xyz)
					vtxset.push_back(vec3f(z, x, y)*scale+shift);
				else
					vtxset.push_back(vec3f(x, y, z)*scale+shift);
		} else

			if (buf[0] == 'v' && buf[1] == 't') {
				double x, y;
				sscanf(buf + 3, "%lf%lf", &x, &y);

				texset.push_back(vec2f(x, y));
			}
			else
			if (buf[0] == 'f' && buf[1] == ' ') {
				int id0, id1, id2, id3=0;
				int tid0, tid1, tid2, tid3=0;
				bool quad = false;

				int count = sscanf(buf+2, "%d/%d", &id0, &tid0);
				char *nxt = strchr(buf+2, ' ');
				sscanf(nxt+1, "%d/%d", &id1, &tid1);
				nxt = strchr(nxt+1, ' ');
				sscanf(nxt+1, "%d/%d", &id2, &tid2);

				nxt = strchr(nxt+1, ' ');
				if (nxt != NULL && nxt[1] >= '0' && nxt[1] <= '9') {// quad
					if (sscanf(nxt+1, "%d/%d", &id3, &tid3))
						quad = true;
				}

				id0--, id1--, id2--, id3--;
				tid0--, tid1--, tid2--, tid3--;

				triset.push_back(tri3f(id0, id1, id2));
				if (count == 2) {
					ttriset.push_back(tri3f(tid0, tid1, tid2));
				}

				if (quad) {
					triset.push_back(tri3f(id0, id2, id3));
					if (count == 2)
						ttriset.push_back(tri3f(tid0, tid2, tid3));
				}
			}
	}
	fclose(fp);

	if (triset.size() == 0 || vtxset.size() == 0)
		return false;

	numVtx = vtxset.size();
	vtxs = new vec3f[numVtx];
	for (unsigned int i=0; i<numVtx; i++)
		vtxs[i] = vtxset[i];

	numTex = texset.size();
	if (numTex == 0)
		texs = NULL;
	else {
		texs = new vec2f[numTex];
		for (unsigned int i = 0; i < numTex; i++)
			texs[i] = texset[i];
	}

	numTri = triset.size();
	tris = new tri3f[numTri];
	for (unsigned int i=0; i<numTri; i++)
		tris[i] = triset[i];

	int numTTri = ttriset.size();
	if (numTTri == 0)
		ttris = NULL;
	else {
		ttris = new tri3f[numTTri];
		for (unsigned int i = 0; i < numTTri; i++)
			ttris[i] = ttriset[i];
	}

	return true;
}


void initObjs(char *path, int stFrame)
{
	unsigned int numVtx=0, numTri=0, numTex=0;
	vec3f *vtxs = NULL;
	tri3f *tris = NULL;
	vec2f *texs = NULL;
	tri3f *ttris = NULL;

	double scale = 1.;
	vec3f shift(0, 0, 0);

	char buff[512];

#ifdef VEC_CLOTH
	//sprintf(buff, "E:\\work\\cudaCloth-5.5\\meshes\\qwoman2\\body000.obj", sidx);
	//sprintf(buff, "E:\\temp4\\kkkk-cpu\\0000_00zzz.obj");
	//sprintf(buff, "E:\\data\\man_kneeling_sequence\\man_kneels00.obj");
	sprintf(buff, "E:\\data\\man_kneeling_5split\\body000.obj");

#else
#ifdef WIN32
	sprintf(buff, "%s\\%04d_00ob.obj", path, stFrame);
#else
	sprintf(buff, "%s/%04d_ob.obj", path, stFrame);
#endif
#endif

	if (readobjfile(buff, numVtx, numTri, tris, vtxs, scale, shift, false, texs, ttris, numTex)) {
		//printf("loading %s ...\n", buff);

		lions[0] = new mesh(numVtx, numTri, tris, vtxs, texs, ttris, numTex);
		lions[0]->updateNrms();
		printf("Read obj file don (#tri=%d, #vtx=%d)\n", numTri, numVtx);
	}
	else
	for (int idx=0; idx<16; idx++) {
#ifdef WIN32
		sprintf(buff, "%s\\obs_%02d.obj", path, idx);
#else
		sprintf(buff, "%s/obs_%02d.obj", path, idx);
#endif
		//printf("loading %s ...\n", buff);

		if (readobjfile(buff, numVtx, numTri, tris, vtxs, scale, shift, false, texs, ttris, numTex)) {
			lions[idx] = new mesh(numVtx, numTri, tris, vtxs, NULL, NULL, 0);
			lions[idx]->updateNrms();
			printf("Read obj file don (#tri=%d, #vtx=%d)\n", numTri, numVtx);
		}
	}
}

void initCloth(const char *cfile)
{
	unsigned int numVtx = 0, numTri = 0, numTex = 0;
	vec3f *vtxs = NULL;
	tri3f *tris = NULL;
	vec2f *texs = NULL;
	tri3f *ttris = NULL;

	double scale = 1.f;
	vec3f shift;
	int idx = 0;

	if (readobjfile(cfile, numVtx, numTri, tris, vtxs, scale, shift, false, texs, ttris, numTex)) {
		cloths[idx] = new mesh(numVtx, numTri, tris, vtxs, texs, ttris, numTex);
		g_box += cloths[idx]->bound();
		cloths[idx]->updateNrms();
		printf("Read cloth from obj file (#tri=%d, #vtx=%d)\n", numTri, numVtx);
		cout << g_box.getMin() << endl;
		cout << g_box.getMax() << endl;

		cout << "w =" << g_box.width() << ", h =" << g_box.height() << ", d =" << g_box.depth() << endl;
	}
}

void initModel(const char *cfile)
{
	initCloth(cfile);
}


void quitModel()
{
	for (int i=0; i<16; i++)
		if (cloths[i])
			delete cloths[i];
	for (int i=0; i<16; i++)
		if (lions[i])
			delete lions[i];
}

extern void beginDraw(BOX &);
extern void endDraw();

void drawOther();
void drawBVH(int level);

void drawModel(bool tri, bool pnt, bool edge, bool re, int level)
{
	if (!g_box.empty())
		beginDraw(g_box);

	drawOther();
	drawBVH(-1);
	drawBVH(level);

	if (!g_box.empty())
		endDraw();
}


void
exportCloth(const char *ipath, const char *opath)
{
	FILE *ifp = fopen(ipath, "rt");
	FILE *ofp = fopen(opath, "wt");

	mesh *m = cloths[0];
	int num = m->getNbVertices();
	vec3f *vtxs = m->getVtxs();

	for (unsigned int i = 0; i<num; i++) {
		vec3f pt = vtxs[i];

		fprintf(ofp, "v %lf %lf %lf\n", pt.x, pt.y, pt.z);
	}
	fprintf(ofp, "\n");

	char buf[1024];
	while (fgets(buf, 1024, ifp)) {
		if (buf[0] == 'v' && buf[1] == ' ') {
			continue;
		}
		fputs(buf, ofp);
	}

	fclose(ifp);
	fclose(ofp);
}
