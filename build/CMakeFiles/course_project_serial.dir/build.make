# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.14

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:


#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:


# Remove some rules from gmake that .SUFFIXES does not remove.
SUFFIXES =

.SUFFIXES: .hpux_make_needs_suffix_list


# Suppress display of executed commands.
$(VERBOSE).SILENT:


# A target that is always out of date.
cmake_force:

.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /usr/bin/cmake

# The command to remove a file.
RM = /usr/bin/cmake -E remove -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/thomaskahle/course-project-thomaskAHLE

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/thomaskahle/course-project-thomaskAHLE/build

# Include any dependencies generated for this target.
include CMakeFiles/course_project_serial.dir/depend.make

# Include the progress variables for this target.
include CMakeFiles/course_project_serial.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/course_project_serial.dir/flags.make

CMakeFiles/course_project_serial.dir/source/course-project-serial.cpp.o: CMakeFiles/course_project_serial.dir/flags.make
CMakeFiles/course_project_serial.dir/source/course-project-serial.cpp.o: ../source/course-project-serial.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/thomaskahle/course-project-thomaskAHLE/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/course_project_serial.dir/source/course-project-serial.cpp.o"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/course_project_serial.dir/source/course-project-serial.cpp.o -c /home/thomaskahle/course-project-thomaskAHLE/source/course-project-serial.cpp

CMakeFiles/course_project_serial.dir/source/course-project-serial.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/course_project_serial.dir/source/course-project-serial.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/thomaskahle/course-project-thomaskAHLE/source/course-project-serial.cpp > CMakeFiles/course_project_serial.dir/source/course-project-serial.cpp.i

CMakeFiles/course_project_serial.dir/source/course-project-serial.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/course_project_serial.dir/source/course-project-serial.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/thomaskahle/course-project-thomaskAHLE/source/course-project-serial.cpp -o CMakeFiles/course_project_serial.dir/source/course-project-serial.cpp.s

# Object files for target course_project_serial
course_project_serial_OBJECTS = \
"CMakeFiles/course_project_serial.dir/source/course-project-serial.cpp.o"

# External object files for target course_project_serial
course_project_serial_EXTERNAL_OBJECTS =

course_project_serial: CMakeFiles/course_project_serial.dir/source/course-project-serial.cpp.o
course_project_serial: CMakeFiles/course_project_serial.dir/build.make
course_project_serial: CMakeFiles/course_project_serial.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/thomaskahle/course-project-thomaskAHLE/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable course_project_serial"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/course_project_serial.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/course_project_serial.dir/build: course_project_serial

.PHONY : CMakeFiles/course_project_serial.dir/build

CMakeFiles/course_project_serial.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/course_project_serial.dir/cmake_clean.cmake
.PHONY : CMakeFiles/course_project_serial.dir/clean

CMakeFiles/course_project_serial.dir/depend:
	cd /home/thomaskahle/course-project-thomaskAHLE/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/thomaskahle/course-project-thomaskAHLE /home/thomaskahle/course-project-thomaskAHLE /home/thomaskahle/course-project-thomaskAHLE/build /home/thomaskahle/course-project-thomaskAHLE/build /home/thomaskahle/course-project-thomaskAHLE/build/CMakeFiles/course_project_serial.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/course_project_serial.dir/depend
