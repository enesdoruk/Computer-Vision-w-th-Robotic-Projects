cmake_minimum_required(VERSION 3.0.2)
project(classification)

find_package(catkin REQUIRED COMPONENTS
  rospy
  std_msgs
)


catkin_package(
#  INCLUDE_DIRS include
#  LIBRARIES bag2video
#  CATKIN_DEPENDS rospy std_msgs
#  DEPENDS system_lib
)


include_directories(
# include
  ${catkin_INCLUDE_DIRS}
)

catkin_install_python(PROGRAMS src/weather_cls.py
	DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

