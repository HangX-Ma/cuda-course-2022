#pragma once

#include <float.h>
#include <stdlib.h>
#include "real.hpp"
#include "vec3f.h"
#include "cmesh.h"
#include "box.h"

#include <vector>
using namespace std;

#define MAX(a,b)	((a) > (b) ? (a) : (b))
#define MIN(a,b)	((a) < (b) ? (a) : (b))

class bvh;
class bvh_node;

class bvh_node {
	int _flag;
	BOX _box;
	int _child; // >=0 leaf with tri_id, <0 left & right
	int _parent;

	void setParent(int p) { _parent = p; }

public:
	bvh_node() {
		_flag = 0;
		_child = 0;
		_parent = 0;
	}

	~bvh_node() {
		NULL;
	}

	void query(BOX &bx, std::vector<int> &rets, int id=-1) {
		if (!_box.overlaps(bx))
			return;

		if (isLeaf()) {
			if (id != -1 && id == _child)
				;
			else
				rets.push_back(_child);

			return;
		}

		left()->query(bx, rets);
		right()->query(bx, rets);
	}

	void construct(unsigned int id);
	void construct(unsigned int *lst, unsigned int num);

	void visualize(int level);
	void refit();
	void resetParents(bvh_node *root);

	FORCEINLINE BOX &box() { return _box; }
	FORCEINLINE bvh_node *left() { return this - _child; }
	FORCEINLINE bvh_node *right() { return this - _child + 1; }
	FORCEINLINE int triID() { return _child; }
	FORCEINLINE int isLeaf() { return _child >= 0; }
	FORCEINLINE int parentID() { return _parent; }

	FORCEINLINE void getLevel(int current, int &max_level) {
		if (current > max_level)
			max_level = current;

		if (isLeaf()) return;
		left()->getLevel(current+1, max_level);
		right()->getLevel(current+1, max_level);
	}

	FORCEINLINE void getLevelIdx(int current, unsigned int *idx) {
		idx[current]++;

		if (isLeaf()) return;
		left()->getLevelIdx(current+1, idx);
		right()->getLevelIdx(current+1, idx);
	}


	friend class bvh;
};

class mesh;

class bvh {
	int _num; // all face num
	bvh_node *_nodes;

	void construct(std::vector<mesh*> &);
	void refit();
	void reorder(); // for breath-first refit
	void resetParents();

public:
	bvh(std::vector<mesh*> &ms);

	~bvh() {
		if (_nodes)
			delete [] _nodes;
	}
	
	bvh_node *root() { return _nodes; }

	void refit(std::vector<mesh*> &ms);

	void query(BOX &bx, std::vector<int> &rets, int id=-1) {
		rets.clear();
		root()->query(bx, rets, id);
	}

	void visualize(int);
};