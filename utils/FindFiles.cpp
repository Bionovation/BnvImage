#include "FindFiles.h"

#include <iostream>
#include <fstream>
#include <stdlib.h>  
#include <stdio.h> 

#ifdef _WIN32

#include <Windows.h>
#include <direct.h>

using namespace std;

std::vector<std::string> FindFiles::findFiles(std::string lpPath, std::string secName /*= ".*" */ )
{
	std::string szFind;
	std::string szFile;

    WIN32_FIND_DATAA FindFileData;

	szFind = lpPath;
	szFind = szFind + "\\*";
	szFind = szFind + secName;

    HANDLE hFind=::FindFirstFileA(szFind.c_str(),&FindFileData);

    if(INVALID_HANDLE_VALUE == hFind)
    {
        std::cout << "Empty folder!" << std::endl;
        return std::vector<std::string>();
    }

    do
    {
        if(FindFileData.dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY)
        {
            if(FindFileData.cFileName[0]!='.')
            {
				szFile = szFile + lpPath + "\\" + FindFileData.cFileName;
                findFiles(szFile);
            }
        }
        else
        {
            //if ( szFile[0] )
            {
                std::string filePath = lpPath;
                filePath += "\\";
                filePath += FindFileData.cFileName;
                file_lists.push_back(filePath);
            }
           /* else
            {
                std::string filePath = szFile;
                filePath += FindFileData.cFileName;
                file_lists.push_back(filePath);
            }*/
        }

    }while(::FindNextFileA(hFind,&FindFileData));

    ::FindClose(hFind);
    return file_lists;
}


std::vector<std::string> FindFiles::findOneLevelFileNames(string lpPath, string secName)
{
	std::vector<std::string> fileNames;

	std::string szFind;
	std::string szFile;

	WIN32_FIND_DATAA FindFileData;

	szFind = lpPath;
	szFind = szFind + "\\*";
	szFind = szFind + secName;

	HANDLE hFind = ::FindFirstFileA(szFind.c_str(), &FindFileData);

	if (INVALID_HANDLE_VALUE == hFind)
	{
		std::cout << "Empty folder!" << std::endl;
		return std::vector<std::string>();
	}

	do
	{
		if (FindFileData.dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY)
		{
			
		}
		else
		{
			fileNames.push_back(FindFileData.cFileName);
		}

	} while (::FindNextFileA(hFind, &FindFileData));

	::FindClose(hFind);
	return fileNames;
}

std::vector<std::string> FindFiles::findOneLevelFolderNames(std::string lpPath, std::string secName /*= ".*" */)
{
	std::vector<std::string> fileNames;
	std::string szFind;
	std::string szFile;

	WIN32_FIND_DATAA FindFileData;

	szFind = lpPath;
	szFind = szFind + "\\*";
	szFind = szFind + secName;

	HANDLE hFind = ::FindFirstFileA(szFind.c_str(), &FindFileData);

	if (INVALID_HANDLE_VALUE == hFind)
	{
		std::cout << "Empty folder!" << std::endl;
		return std::vector<std::string>();
	}

	do
	{
		if (FindFileData.dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY)
		{
			if (FindFileData.cFileName[0] != '.')
			{
				fileNames.push_back(FindFileData.cFileName);
			}
		}
		else
		{
			
		}

	} while (::FindNextFileA(hFind, &FindFileData));

	::FindClose(hFind);
	return fileNames;
}

int FindFiles::findAndRevert(string lpPath, string secName /*= ".*"*/)
{
	return 0;
}

#else

#include <dirent.h>
#include <string.h>
#include <queue>

std::vector<std::string> FindFiles::findFiles( const char *lpPath, const char *secName /*= ".*" */ )
{
    (void)secName;

    std::vector<std::string> result;
    std::queue<std::string> queue;
    std::string dirname;

    DIR *dir;
    if ( !(dir = opendir ( lpPath )) ) {
        return result;
    }
    queue.push( lpPath );

    struct dirent *ent;
    while ( !queue.empty() )
    {

        dirname = queue.front();
        dir = opendir( dirname.c_str() );
        queue.pop();
        if ( !dir ) { continue; }

        while( ent = readdir( dir ) )
        {

            if ( strcmp(".", ent->d_name) == 0 || strcmp("..", ent->d_name) == 0 )
            {
                continue;
            }
            if ( ent->d_type == DT_DIR )
            {
                queue.push( dirname+"/"+ent->d_name );
            }
            else
            {
                result.push_back( dirname+"/"+ent->d_name );
            }

        }

        closedir( dir );

    }

    return result;
}

#endif
