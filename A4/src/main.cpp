#include <iostream>
#include <string>
#include <vector>
#include "optionparser.h"

enum  optionIndex { UNKNOWN, HELP, FILE_PATH, NOTE };
const option::Descriptor usage[] = {
    {UNKNOWN, 0, "", "", option::Arg::None, "USAGE: program [options]\n\n"
                                            "Options:" },
    {HELP, 0, "h", "help", option::Arg::None, "  --help, -h \tPrint usage and exit." },
    {FILE_PATH, 0, "f", "file", option::Arg::None, "  --file, -f \tOjbect file path." },
    {NOTE, 0, "", "", option::Arg::None, "\nNOTE: (.obj) file needs to be determined by yourself."},
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