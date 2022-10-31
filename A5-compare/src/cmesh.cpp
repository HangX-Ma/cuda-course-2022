#if defined(WIN32)
#define WIN32_LEAN_AND_MEAN
#  include <windows.h>
#endif
#include <GL/gl.h>

#include <stdio.h>
#include <string.h>

//#include "mesh_defs.h"
#include "cmesh.h"
#include "box.h"

#include <set>
using namespace std;

// for fopen
#pragma warning(disable: 4996)

inline vec3f update(vec3f &v1, vec3f &v2, vec3f &v3)
{
	vec3f s = (v2-v1);
	return s.cross(v3-v1);
}

inline vec3f
update(tri3f &tri, vec3f *vtxs)
{
	vec3f &v1 = vtxs[tri.id0()];
	vec3f &v2 = vtxs[tri.id1()];
	vec3f &v3 = vtxs[tri.id2()];

	return update(v1, v2, v3);
}

void mesh::update(matrix3f &trf, vec3f &shift)
{
	if (_first) {
		_first = false;
		memcpy(_ivtxs, _vtxs, sizeof(vec3f)*_num_vtx);
	}

	for (unsigned int i=0; i<_num_vtx; i++)
		_vtxs[i] = _ivtxs[i]*trf + shift;

}

vec3f mesh::faceNrm(int fid)
{
	vec3f n = ::update(_tris[fid], _vtxs);
	n.normalize();

	return n;
}

void mesh::updateNrms()
{
	for (unsigned int i=0; i<_num_vtx; i++)
		_nrms[i] = vec3f::zero();

	for (unsigned int i=0; i<_num_tri; i++) {
		vec3f n = ::update(_tris[i], _vtxs);
		n.normalize();

		_nrms[_tris[i].id0()] += n;
		_nrms[_tris[i].id1()] += n;
		_nrms[_tris[i].id2()] += n;
	}

	for (unsigned int i=0; i<_num_vtx; i++)
		_nrms[i].normalize();
}

bool mesh::load(FILE *fp)
{
	fread(_vtxs, sizeof(vec3f), _num_vtx,  fp);
	
	// skip the _vels ...
	fread(_nrms, sizeof(vec3f), _num_vtx, fp);

	return true;
}

void initRedMat(int side)
{
	GLfloat matAmb[4] =    {1.0, 1.0, 1.0, 1.0};
	GLfloat matDiff[4] =   {1.0, 0.1, 0.2, 1.0};
	GLfloat matSpec[4] =   {1.0, 1.0, 1.0, 1.0};
	glMaterialfv(side, GL_AMBIENT, matAmb);
	glMaterialfv(side, GL_DIFFUSE, matDiff);
	glMaterialfv(side, GL_SPECULAR, matSpec);
	glMaterialf(side, GL_SHININESS, 600.0);
}

void initBlueMat(int side)
{
	GLfloat matAmb[4] =    {1.0, 1.0, 1.0, 1.0};
	GLfloat matDiff[4] =   {0.0, 1.0, 1.0, 1.0};
	GLfloat matSpec[4] =   {1.0, 1.0, 1.0, 1.0};
	glMaterialfv(side, GL_AMBIENT, matAmb);
	glMaterialfv(side, GL_DIFFUSE, matDiff);
	glMaterialfv(side, GL_SPECULAR, matSpec);
	glMaterialf(side, GL_SHININESS, 60.0);
}

void initYellowMat(int side)
{
	GLfloat matAmb[4] =    {1.0, 1.0, 1.0, 1.0};
	GLfloat matDiff[4] =   {1.0, 1.0, 0.0, 1.0};
	GLfloat matSpec[4] =   {1.0, 1.0, 1.0, 1.0};
	glMaterialfv(side, GL_AMBIENT, matAmb);
	glMaterialfv(side, GL_DIFFUSE, matDiff);
	glMaterialfv(side, GL_SPECULAR, matSpec);
	glMaterialf(side, GL_SHININESS, 60.0);
}

void initGrayMat(int side)
{
	GLfloat matAmb[4] =    {1.0, 1.0, 1.0, 1.0};
	GLfloat matDiff[4] =   {0.5, 0.5, 0.5, 1.0};
	GLfloat matSpec[4] =   {1.0, 1.0, 1.0, 1.0};
	glMaterialfv(side, GL_AMBIENT, matAmb);
	glMaterialfv(side, GL_DIFFUSE, matDiff);
	glMaterialfv(side, GL_SPECULAR, matSpec);
	glMaterialf(side, GL_SHININESS, 60.0);
}

void setMat(vec3f &cr)
{
	GLfloat front[4] = { cr[0], cr[1], cr[2], 1 };
	GLfloat back[4] = { cr[0], cr[1], cr[2], 1 };
	glMaterialfv(GL_FRONT, GL_AMBIENT_AND_DIFFUSE, front);
	glMaterialfv(GL_BACK, GL_AMBIENT_AND_DIFFUSE, back);
}

void mesh::display(bool t, bool p, bool e, int level, bool rigid,
	set<int> &ids, vector<int> &vids, int id)
{
	glPointSize(10.f);

	glEnable(GL_LIGHTING);
	initRedMat(GL_FRONT);
	initRedMat(GL_BACK);

	glShadeModel(GL_SMOOTH);
	glEnableClientState(GL_VERTEX_ARRAY);
	glEnableClientState(GL_NORMAL_ARRAY);
	glVertexPointer(3, GL_DOUBLE, sizeof(double) * 3, _vtxs);
	glNormalPointer(GL_DOUBLE, sizeof(double) * 3, _nrms);

	glDrawElements(GL_TRIANGLES, _num_tri * 3, GL_UNSIGNED_INT, _tris);
	glDisableClientState(GL_VERTEX_ARRAY);
	glDisableClientState(GL_NORMAL_ARRAY);
}


mesh::mesh(unsigned int numVtx, unsigned int numTri, tri3f *tris, vec3f *vtxs, vec2f *texs, tri3f *ttris, unsigned int numTex)
{
	_first = true;

	_num_vtx = numVtx;
	_num_tri = numTri;

	_tris = tris;
	_vtxs = vtxs;
	_ivtxs = new vec3f[numVtx];
	_ovtxs = new vec3f[numVtx];
	_fflags = NULL;
	_vflags = NULL;

	_ttris = ttris;
	_uvs = texs;
	_num_uv = numTex;

	_nrms = new vec3f[numVtx];
	_bxs = new BOX[numTri];
	_areas = new double[numTri];

	calcAreas(texs, ttris);
	updateBxs();
}

mesh::~mesh()
{
	delete [] _tris;
	delete [] _vtxs;
	delete [] _ivtxs;
	delete [] _ovtxs;
	delete [] _nrms;
	delete[] _bxs;
	if (_fflags) delete[] _fflags;
	if (_vflags) delete[] _vflags;
}

#include "vec3f.h"


void beginDraw(BOX &bx)
{
	glMatrixMode( GL_MODELVIEW );
	glPushMatrix();

	vec3f pt=bx.center();
	double len = bx.height()+bx.depth()+bx.width();
	double sc = 6.0/len;

	//glRotatef(-90, 0, 0, 1);
	glScalef(sc, sc, sc);
	glTranslatef(-pt.x, -pt.y, -pt.z);
}

void endDraw()
{
	glPopMatrix();
}

extern mesh *cloths[];
extern int maxLevel;
extern std::vector<REAL> gIntensity[];
extern int currentPass;

void drawTriangle(int tid)
{
	mesh *m = cloths[0];
	if (m == NULL)
		return;

	tri3f &t = m->_tris[tid];

	vec3f &a = m->_vtxs[t.id0()];
	vec3f &b = m->_vtxs[t.id1()];
	vec3f &c = m->_vtxs[t.id2()];
	vec3f &na = m->_nrms[t.id0()];
	vec3f &nb = m->_nrms[t.id1()];
	vec3f &nc = m->_nrms[t.id2()];

	if (currentPass == -1) {
		glEnable(GL_LIGHTING);
		glBegin(GL_TRIANGLES);
		glNormal3dv(na.v);
		glVertex3dv(a.v);
		glNormal3dv(nb.v);
		glVertex3dv(b.v);
		glNormal3dv(nc.v);
		glVertex3dv(c.v);
		glEnd();
		glDisable(GL_LIGHTING);
	}
	else {
		float tt = gIntensity[currentPass][tid];
		GLfloat matDiff[4] = { 1.0, 1.0, 0.0, 1.0 };
		matDiff[1] = 1 - tt;
		matDiff[0] = tt;

		glEnable(GL_LIGHTING);
		glMaterialfv(GL_FRONT, GL_DIFFUSE, matDiff);
		glBegin(GL_TRIANGLES);
		glNormal3dv(na.v);
		glVertex3dv(a.v);
		glNormal3dv(nb.v);
		glVertex3dv(b.v);
		glNormal3dv(nc.v);
		glVertex3dv(c.v);
		glEnd();

		glDisable(GL_LIGHTING);
	}
}

void drawTriangle(int tid, int flag)
{
	mesh *m = cloths[0];
	if (m == NULL)
		return;

	tri3f &t = m->_tris[tid];

	vec3f &a = m->_vtxs[t.id0()];
	vec3f &b = m->_vtxs[t.id1()];
	vec3f &c = m->_vtxs[t.id2()];
	vec3f &na = m->_nrms[t.id0()];
	vec3f &nb = m->_nrms[t.id1()];
	vec3f &nc = m->_nrms[t.id2()];

	if (flag == 0) {
		glEnable(GL_LIGHTING);
		glBegin(GL_TRIANGLES);
		glNormal3dv(na.v);
		glVertex3dv(a.v);
		glNormal3dv(nb.v);
		glVertex3dv(b.v);
		glNormal3dv(nc.v);
		glVertex3dv(c.v);
		glEnd();
		glDisable(GL_LIGHTING);
	}
	else {
#if 0
		glDisable(GL_LIGHTING);

		float t = flag / float(maxLevel);
		glColor3f(1-t, t, 0.0);

		glBegin(GL_TRIANGLES);
		glNormal3dv(na.v);
		glVertex3dv(a.v);
		glNormal3dv(nb.v);
		glVertex3dv(b.v);
		glNormal3dv(nc.v);
		glVertex3dv(c.v);
		glEnd();
#else
		glEnable(GL_LIGHTING);

		float t = flag / float(maxLevel);
		GLfloat matDiff[4] = { 1.0, 1.0, 0.0, 1.0 };
		matDiff[0] = 1 - t;
		matDiff[1] = t;

		glMaterialfv(GL_FRONT, GL_DIFFUSE, matDiff);
		glBegin(GL_TRIANGLES);
		glNormal3dv(na.v);
		glVertex3dv(a.v);
		glNormal3dv(nb.v);
		glVertex3dv(b.v);
		glNormal3dv(nc.v);
		glVertex3dv(c.v);
		glEnd();

#endif

		glDisable(GL_LIGHTING);
	}
}


#include "GL/glut.h"

void
aabb::visualize()
{
#if 0
	glColor3f(0, 1, 0);
	glLineWidth(3.0);
#else
	glColor3f(1.0f, 1.0f, 1.0f);
#endif

	glPushMatrix();
	::vec3f org = center();
	glTranslatef(org[0], org[1], org[2]);

	float w = width();
	float h = height();
	float d = depth();

	glScalef(w, h, d);
	glutWireCube(1.f);
	glPopMatrix();
}

#include "tmbvh.hpp"

void
bvh::visualize(int level)
{
	glDisable(GL_LIGHTING);
	if (_nodes)
		_nodes[0].visualize(level);
	glEnable(GL_LIGHTING);
}

