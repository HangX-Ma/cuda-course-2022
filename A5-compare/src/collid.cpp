#include "cmesh.h"
#include <set>
#include <iostream>
#include <stdio.h>

using namespace std;
#include "mat3f.h"
#include "box.h"
#include "tmbvh.hpp"

#define MAX_CD_PAIRS 4096*1024

extern mesh *cloths[16];
extern mesh *lions[16];

extern vector<int> vtx_set;
extern set<int> cloth_set;
extern set<int> lion_set;
static bvh *bvhCloth = NULL;
static bvh *bvhBody = NULL;

bool findd;

#include <chrono>
#include "kernel.h"

#define        TIMING_BEGIN \
        {auto time_start = std::chrono::system_clock::now();

#define        TIMING_END(message) \
        {auto time_end = std::chrono::system_clock::now();\
        auto  time_duration = \
        std::chrono::duration_cast<std::chrono::microseconds>(time_end - time_start);\
        printf("%s: %2.5f milliseconds\n", (message), static_cast<double>(time_duration.count()/1000.0));}}

#define DO_GPU (0)



// CPU with BVH
void buildBVH()
{

	TIMING_BEGIN
	static std::vector<mesh *> meshes;

		if (bvhCloth == NULL)
		{
			for (int i = 0; i < 16; i++)
				if (cloths[i] != NULL)
					meshes.push_back(cloths[i]);

			bvhCloth = new bvh(meshes);
		}

	bvhCloth->refit(meshes);
	TIMING_END("bvh done...")
}

void drawBVH(int level)
{
	if (bvhCloth == NULL) return;
	bvhCloth->visualize(level);
}


int maxLevel = 60;
std::vector<std::vector<int>> gAdjInfo;
std::vector<REAL> gIntensity[2];
int currentPass = -1;
std::vector<int> gSources;


void doPropagateGPU() {
	mesh *mc = cloths[0];
	int num = mc->getNbFaces();

	TIMING_BEGIN
	doPropagateKernel(currentPass, num);

	for (int i = 0; i < gSources.size(); i++) {
		gIntensity[currentPass][gSources[i]] = 1.0;
	}
	currentPass = 1 - currentPass;
	TIMING_END("propogating...")
}




void doPropogate()
{
	int prevPass = currentPass;
	currentPass = 1 - currentPass;

	mesh *mc = cloths[0];
	int num = mc->getNbFaces();

	TIMING_BEGIN
		for (int i = 0; i < num; i++) {
			std::vector<int> &adjs = gAdjInfo[i];
			gIntensity[currentPass][i] = 0.6 * gIntensity[prevPass][i];
			for (int j = 0; j < adjs.size(); j++) {
				int tj = adjs[j];
				gIntensity[currentPass][i] += gIntensity[prevPass][tj];
			}
			// printf("%u ", adjs.size());
			gIntensity[currentPass][i] /= REAL(adjs.size() + 1);
			gIntensity[currentPass][i] += HEAT_TRANSFER_SPEED * gIntensity[prevPass][i];
			gIntensity[currentPass][i] = fmin(gIntensity[currentPass][i], 1.0f);
		}

	for (int i = 0; i < gSources.size(); i++) {
		gIntensity[currentPass][gSources[i]] = 1.0;
	}
	TIMING_END("propogating...")
}

extern void buildIt();

void doIt()
{
	if (currentPass == -1) {
		currentPass = 1;
		buildIt();

		gSources.push_back(100);
		gSources.push_back(10);
		gSources.push_back(200);

		mesh *mc = cloths[0];
		int num = mc->getNbFaces();
		gIntensity[0].resize(num);
		gIntensity[1].resize(num);

		for (int i = 0; i < num; i++) {
			gIntensity[currentPass][i] = 0;
		}

		for (int i = 0; i < gSources.size(); i++) {
			gIntensity[currentPass][gSources[i]] = 1.0;
		}

		#if DO_GPU
		doGPUInit(num, currentPass);
		#endif
	}
	else {
		#if DO_GPU
		doPropagateGPU();
		#else
		doPropogate();
		#endif
	}
}

void buildIt()
{
	mesh *mc = cloths[0];
	int num = mc->getNbFaces();
	gAdjInfo.clear();
	gAdjInfo.resize(num);

	TIMING_BEGIN
	for (int i = 0; i < num; i++) {
		BOX bx = mc->getTriBox(i);
		bvhCloth->query(bx, gAdjInfo[i], i);
	}
	TIMING_END("build adj info")
}
