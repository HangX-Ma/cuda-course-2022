#pragma once

#include <stdio.h>
#include "vec3f.h"
#include "mat3f.h"

#include "tri.h"
#include "edge.h"
#include "box.h"

#include <set>
#include <vector>
using namespace std;

class mesh {
public:
	unsigned int _num_vtx;
	unsigned int _num_tri;
	unsigned int _num_uv;

	tri3f *_tris;
	BOX *_bxs;
	double *_areas;

	// used by time integration
	vec3f *_vtxs;
	vec3f *_ivtxs; // initial positions
	vec3f *_ovtxs; // previous positions
	vec3f *_nrms;

	tri3f *_ttris;
	vec2f *_uvs;

	int *_fflags;
	int *_vflags;

	bool _first;

	mesh(unsigned int numVtx, unsigned int numTri, tri3f *tris, vec3f *vtxs, vec2f *texs, tri3f *ttris, unsigned int numTex);
	~mesh();

	unsigned int getNbVertices() const { return _num_vtx; }
	unsigned int getNbFaces() const { return _num_tri; }
	unsigned int getNbUVs() const { return _num_uv; }

	vec2f *getUVs() const { return _uvs; }
	vec3f *getVtxs() const { return _vtxs; }
	vec3f *getOVtxs() const { return _ovtxs;}
	vec3f *getIVtxs() const {return _ivtxs;}
	vec3f *getNrms()  const { return _nrms; }

	BOX getTriBox(int tid) const {
		return _bxs[tid];
	}

	BOX getTriBox2(int tid) const {
		tri3f &a = _tris[tid];
		vec3f p0 = _vtxs[a.id0()];
		vec3f p1 = _vtxs[a.id1()];
		vec3f p2 = _vtxs[a.id2()];

		BOX bx(p0, p1);
		bx += p2;
		return bx;
	}

	void update(matrix3f &, vec3f &);

	void setFFlags(int *);
	void setVFlags(int *);

	// calc norms, and prepare for display ...
	void updateNrms();
	vec3f faceNrm(int fid);

	// really displaying ...
	void display_tex(bool tri, bool pnt, bool edge, int level, bool rigid, set<int>&, vector<int> &, int);
	void display(bool tri, bool pnt, bool edge, int level, bool rigid, set<int>&, vector<int> &, int);


	// load vtxs
	bool load(FILE *fp);

	void calcAreas(vec2f *texs, tri3f *ttris) {
		for (int i = 0; i < _num_tri; i++) {
			if (texs == NULL) {
				_areas[i] = -1;
				continue;
			}

			tri3f &a = ttris[i];
			vec2f &u0 = texs[a.id0()];
			vec2f &u1 = texs[a.id1()];
			vec2f &u2 = texs[a.id2()];

			double tmp = (u1-u0).cross(u2-u0);
			_areas[i] = fabs(tmp)*0.5;
		}
	}

	void updateBxs() {
		for (int i = 0; i < _num_tri; i++) {
			_bxs[i] = getTriBox2(i);
		}
		printf("Boxes updated.\n");
	}

	BOX bound() {
		BOX a;

		for (int i=0; i<_num_vtx; i++)
			a += _vtxs[i];

		return a;
	}
};
