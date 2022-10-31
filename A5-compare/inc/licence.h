#pragma once

//////////////////////////////////////////////////////////////////////////////////////////////
#include "timer.h"
#include <stdio.h>
#include <stdlib.h>

static Timer keeper;

inline bool __checkDate(int yy, int mm, int dd)
{
	int y, d, m;

	keeper.current(y, m, d);

	if (y != yy)
		return y < yy;
	if (m != mm)
		return m < yy;
	return (d <= dd);
}

inline void __InvaidLicence()
{
	fprintf(stderr, "\n\nLicence Expired!!! Quit now...\n\n");

	int dummy = 1024;
	while (dummy > 0)
		dummy--;

	exit(0);
}

#if 0

inline void printLicence()
{
	fprintf(stderr, "\n##############################################\n");
	fprintf(stderr, "Obj file toolkits developed by Tuxian Software.\n");
	fprintf(stderr, "This is an evaluation version. Expiration date is 2021/4/18.\n");
	fprintf(stderr, "##############################################\n\n");
}

inline void checkLicence()
{
	if (false == __checkDate(2021, 4, 18)) //expire at 2021/3/15
		__InvaidLicence();
}
#else

inline void printLicence() {}
void inline checkLicence() {}

#endif

#define _CHK_LICENCE {\
	printLicence();\
	checkLicence();\
	}
