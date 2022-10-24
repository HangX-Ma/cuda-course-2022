#define TINYOBJLOADER_IMPLEMENTATION // define this in only *one* .cc
#include "tiny_obj_loader.h"
#include "optionparser.h"
#include "bvh.h"
#include "obj-viewer.h"

#include <iostream>
#include <string>
#include <vector>

std::uint32_t G_num_objects;
lbvh::triangle_t* G_triangles;
lbvh::vec3f* G_vertices;


struct Arg: public option::Arg
{
  static void printError(const char* msg1, const option::Option& opt, const char* msg2)
  {
    fprintf(stderr, "%s", msg1);
    fwrite(opt.name, opt.namelen, 1, stderr);
    fprintf(stderr, "%s", msg2);
  }

    static option::ArgStatus Unknown(const option::Option& option, bool msg)
    {
        if (msg) printError("Unknown option '", option, "'\n");
        return option::ARG_ILLEGAL;
    }

    static option::ArgStatus Required(const option::Option& option, bool msg)
    {
    if (option.arg != 0)
        return option::ARG_OK;

    if (msg) printError("Option '", option, "' requires an argument\n");
    return option::ARG_ILLEGAL;
    }
};

enum  optionIndex { UNKNOWN, HELP, FILE_PATH, NOTE };
const option::Descriptor usage[] = {
    {UNKNOWN,   0, "",  "",     Arg::None,      "USAGE: program [options]\n\n"
                                                "Options:" },
    {HELP,      0, "",  "help", Arg::None,      "  --help, \tPrint usage and exit." },
    {FILE_PATH, 0, "f", "file", Arg::Required,  "  --file, -f \tOjbect file path." },
    {NOTE,      0, "",  "",     Arg::None,      "\nNOTE: (.obj) file needs to be determined by yourself."},
    {0, 0, 0, 0, 0, 0}
};
 
int main(int argc, char* argv[])
{
    argc-=(argc>0); argv+=(argc>0); // skip program name argv[0] if present
    option::Stats stats(usage, argc, argv);
    std::vector<option::Option> options(stats.options_max);
    std::vector<option::Option> buffer(stats.buffer_max);
    option::Parser parse(usage, argc, argv, &options[0], &buffer[0]);
    
    if (parse.error())
        return 1;
    
    if (options[HELP] || argc == 0) {
        option::printUsage(std::cout, usage);
        return 0;
    }
    
    if (options[FILE_PATH] || argc == 0) {
        // initialize rendering with solid body
        switch_render_mode(true);
        glutInit(&argc, argv);
        glutInitDisplayMode(GLUT_RGBA | GLUT_DOUBLE | GLUT_ALPHA | GLUT_DEPTH);
        glutInitWindowSize(960, 720);
        glutInitWindowPosition(0, 0);
        glutCreateWindow("Justin Tennant OBJ Visualizer");

        init();
        std::string objFilePath(options[FILE_PATH].arg);
        lbvh::BVH objBVH;
        objBVH.loadObj(objFilePath);
        objBVH.construct();

        G_num_objects = objBVH.getOjbectNum();
        G_triangles = objBVH.getTriangleList();
        G_vertices = objBVH.getVerticeList();

        glutDisplayFunc(display);
        glutReshapeFunc(reshape);
        glutSpecialFunc(arrow_keys);
        glutKeyboardFunc(keyboard);
        glutMainLoop();

        return 0;
    }
    
    /* Deal with special conditions */
    for (option::Option* opt = options[UNKNOWN]; opt; opt = opt->next()) {
        std::cout << "Unknown option: " << std::string(opt->name, opt->namelen) << "\n";
    }
    
    for (int i = 0; i < parse.nonOptionsCount(); ++i) {
        std::cout << "Non-option #" << i << ": " << parse.nonOption(i) << "\n";
    }
}