#pragma once
#include <opencv2\opencv.hpp>

namespace pu
{
	/// <summary>
	/// Wraps (arbitrary) phase value. 
	/// </summary>
	/// <param name="phase">
	/// Phase to wrap, arbitrary value, in radians.
	/// </param>
	/// <returns>
	/// Wrapped phase in radians in range [-PI, PI].
	/// </returns>
	float Wrap(float phase);


	/// <summary>
	/// Wraps whole phase image. 
	/// </summary>
	/// <param name="phase">
	/// Image to wrap, 1 channel, floating point, arbitrary values.
	/// </param>
	/// <param name="normalize">
	/// [default = true] Whether to normalize values to [0, 1] range 
	/// or not.
	/// </param>
	/// <returns>
	/// Wrapped phase image, 1 channel, floating point. If normalized 
	/// was set values lay in range [0, 1] otherwise [-PI, PI].
	/// </returns>
	cv::Mat Wrap(const cv::Mat& phase, bool normalize = true);
	
}