#pragma once

#include <time.h>
#include <chrono>

struct Timer {
	Timer() {}


	void current(int &y, int &m, int &d)
	{
		time_t  t;
		tm  *tp;
		t = time(NULL);
		//tp = localtime(&t);
		localtime_r(&t, tp);

#if 0
		printf("%d/%d/%d/n", tp->tm_mon + 1, tp->tm_mday, tp->tm_year + 1900);
		printf("%d:%d:%d/n", tp->tm_hour, tp->tm_min, tp->tm_sec);
#endif

		m = tp->tm_mon + 1;
		d = tp->tm_mday;
		y = tp->tm_year + 1900;
	}
};
