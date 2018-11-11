#pragma once
#include <cmath>
#include <opencv2\opencv.hpp>

namespace pu
{
	/// <summary>
	/// Scales input accordingly so that if it ran in range [srcMin, srcMax] 
	/// it is scaled to run in range [dstMin, dstMax].
	/// </summary>
	/// <param name="value">
	/// Value to scale.
	/// </param>
	/// <param name="srcMin">
	/// Minimum of the range input value lays in.
	/// </param>
	/// <param name="srcMax">
	/// Maximum of the range input value lays in.
	/// </param>
	/// <param name="dstMin">
	/// Minimum of the range output value should lay in, default = 0.
	/// </param>
	/// <param name="dstMax">
	/// Minimum of the range output value should lay in, default = 0.
	/// </param>
	/// <returns></returns>
	float Scale(float value, float srcMin, float srcMax, float dstMin = 0, float dstMax = 1);

	/// <summary>
	/// Converts provided image to displayable.
	/// </summary>
	/// <param name="dstMax">
	/// Image to convert, should be single channel.
	/// </param>
	/// <returns>
	/// Single channel, floating point image which values lay in [0, 1] range.
	/// </returns>
	cv::Mat ToDisplayable(const cv::Mat& m);
}