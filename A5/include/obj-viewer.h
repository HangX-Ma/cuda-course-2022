/**
 * @file obj-viewer.h
 * @author HangX-Ma m-contour@qq.com
 * @brief  object viewer
 * @version 0.1
 * @date 2022-10-10
 * 
 * @copyright Copyright (c) 2022 HangX-Ma(MContour) m-contour@qq.com
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#ifndef __OBJ_VIEWER_H__
#define __OBJ_VIEWER_H__


#include <GL/glut.h>
#include <cmath>
#include <cstring>
#include <iostream>
#include <fstream>
#include <regex>

struct OBJ_COLOR {
    GLfloat red, green, blue;
    OBJ_COLOR() : red(1.0), green(1.0), blue(1.0) {}
};

struct camera {
    GLfloat x, y, z, phi, theta;
    camera() : x(-4.0f), y(2.0f), z(0.0f), phi(0), theta(0) {}
};

void switch_render_mode(bool mode);
void init();
void reshape(int w, int h);
void arrow_keys(int key, int x, int y);
void keyboard(unsigned char key, int x, int y);
void display();
void idle_fem();

#endif