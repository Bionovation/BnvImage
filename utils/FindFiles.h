/*
 *   Find and generate a file list of the folder.
**/

#ifndef FIND_FILES_H
#define FIND_FILES_H

#include <vector>
#include <string>

using std::string;

class FindFiles
{
public:
    FindFiles(){}
    ~FindFiles(){}

    std::vector<std::string> findFiles( string lpPath, string secName = ".*" );
	static std::vector<std::string> findOneLevelFolderNames(string lpPath, string secName = ".*");
	static std::vector<std::string> findOneLevelFileNames(string lpPath, string secName = ".*");
	int findAndRevert(std::string lpPath, std::string secName = ".*");

private:
    std::vector<std::string> file_lists;
};

#endif // FIND_FILES_H
