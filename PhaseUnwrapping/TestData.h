#pragma once
#include <opencv2\opencv.hpp>
#define _USE_MATH_DEFINES
#include <math.h>

namespace pu
{
	constexpr int DEFAULT_TEST_ROWS = 128;
	constexpr int DEFAULT_TEST_COLS = 128;
	constexpr float DEFAULT_TEST_MIN = -M_PI * 10;
	constexpr float DEFAULT_TEST_MAX = M_PI * 10;

	cv::Mat VerticalPlane(
		int rows = DEFAULT_TEST_ROWS,
		int cols = DEFAULT_TEST_COLS,
		float minVal = DEFAULT_TEST_MIN,
		float maxVal = DEFAULT_TEST_MAX);

	cv::Mat HorizontalPlane(
		int rows = DEFAULT_TEST_ROWS,
		int cols = DEFAULT_TEST_COLS,
		float minVal = DEFAULT_TEST_MIN,
		float maxVal = DEFAULT_TEST_MAX);

	cv::Mat ShearPlanes(
		int rows = DEFAULT_TEST_ROWS,
		int cols = DEFAULT_TEST_COLS,
		float minVal = DEFAULT_TEST_MIN,
		float maxVal = DEFAULT_TEST_MAX);

	cv::Mat SpiralShear(
		int rows = DEFAULT_TEST_ROWS,
		int cols = DEFAULT_TEST_COLS,
		float minVal = DEFAULT_TEST_MIN,
		float maxVal = DEFAULT_TEST_MAX);

	cv::Mat Peaks(
		int rows = DEFAULT_TEST_ROWS,
		int cols = DEFAULT_TEST_COLS,
		float minVal = DEFAULT_TEST_MIN,
		float maxVal = DEFAULT_TEST_MAX);

	void AddSaltPepperNoise(cv::Mat & img, float probability = 0.01f);

	void AddRandomNoise(cv::Mat & img, float probability = 0.01f, float magnitude = 1.f);

	namespace
	{
		inline float Peak(float x, float y);
		inline static float Noise();
	}
}
