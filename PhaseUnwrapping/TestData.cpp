#include "TestData.h"
#include "Utils.h"
#include <random>
#include <mutex>

namespace pu
{
	namespace
	{
		float Peak(float x, float y)
		{
			return 3.0f * std::powf(1.0f - x, 2.0f) * std::expf(-std::powf(x, 2.0f) - std::powf(y + 1.0f, 2.0f))
				- 10.0f * (x / 5.0f - std::powf(x, 3.0f) - std::powf(y, 5.0f)) * std::expf(-std::powf(x, 2.0f) - std::powf(y, 2.0f))
				- 1.0f / 3.0f * std::expf(-std::powf(x + 1.0f, 2.0f) - std::powf(y, 2.0f));
		}

		float Noise()
		{
			// Random engine with seed so re-launching program will generate same sequence each time
			static std::mt19937 engine{ 2137 };
			// Uniformly distributed from 0 to 1
			static std::uniform_real_distribution<float> distribution(0.f, 1.f);
			// Bind for ease of use
			static auto noise = std::bind(distribution, engine);

			return noise();
		}
	}

	cv::Mat VerticalPlane(int rows, int cols, float minVal, float maxVal)
	{
		assert(rows > 0 && cols > 0 && "[VerticalPlance] Invalid dimensions");

		// Swap values so the natural order is preserved
		if(minVal > maxVal)
		{
			std::swap(minVal, maxVal);
		}

		cv::Mat img{ rows,cols,CV_32FC1, cv::Scalar(0) };

		// Assign row index to each pixel (scaling will be done later)
		img.forEach<float>([&](float& px, const int* pos) -> void {
			px = static_cast<float>(pos[0]);
		});

		// Scale image so its values run from minVal to maxVal
		cv::normalize(img, img, minVal, maxVal, cv::NORM_MINMAX);

		return img;
	}

	cv::Mat HorizontalPlane(int rows, int cols, float minVal, float maxVal)
	{
		assert(rows > 0 && cols > 0 && "[HorizontalPlane] Invalid dimensions");

		// Swap values so the natural order is preserved
		if(minVal > maxVal)
		{
			std::swap(minVal, maxVal);
		}

		cv::Mat img{ rows,cols,CV_32FC1, cv::Scalar(0) };

		// Assign col index to each pixel (scaling will be done later)
		img.forEach<float>([&](float& px, const int* pos) -> void {
			px = static_cast<float>(pos[1]);
		});

		// Scale image so its values run from minVal to maxVal
		cv::normalize(img, img, minVal, maxVal, cv::NORM_MINMAX);

		return img;
	}

	cv::Mat ShearPlanes(int rows, int cols, float minVal, float maxVal)
	{
		assert(rows > 0 && cols > 0 && "[ShearPlanes] Invalid dimensions");

		// Swap values so the natural order is preserved
		if(minVal > maxVal)
		{
			std::swap(minVal, maxVal);
		}

		cv::Mat img{ rows,cols,CV_32FC1, cv::Scalar(0) };

		// Divide img into left and right planes
		// Left one will increase values downwards
		// Right one will increase values upwards
		// Same value for both plancs will be in the middle row
		img.forEach<float>([&](float& px, const int* pos) -> void {
			int row = pos[0], col = pos[1];
			px = col < img.cols / 2 ? (row - rows / 2) : (rows / 2 - row);
		});

		// Scale image so its values run from minVal to maxVal
		cv::normalize(img, img, minVal, maxVal, cv::NORM_MINMAX);

		return img;
	}

	cv::Mat SpiralShear(int rows, int cols, float minVal, float maxVal)
	{
		assert(rows > 0 && cols > 0 && "[SpiralShear] Invalid dimensions");

		// Swap values so the natural order is preserved
		if(minVal > maxVal)
		{
			std::swap(minVal, maxVal);
		}

		cv::Mat img{ rows,cols,CV_32FC1, cv::Scalar(0) };

		// "Thickness" of spiral arm, delta radius 
		int dR = cols / 8;

		// Y coordinate of each of spirals' semicircles
		int centerRow = rows / 2;

		img.forEach<float>([&](float& px, const int* pos) -> void {
			int row = pos[0], col = pos[1];

			bool isTop = row < (rows / 2);

			// Move center a bit to the right for top half to the left for bottom one
			int centerCol = cols / 2 + (isTop ? 1 : -1) * dR;
			float realMin = isTop ? minVal : maxVal;
			float realMax = isTop ? maxVal : minVal;

			// Squared distance from the center (used to detect whether negative or positive shear should be applied)
			double dist = std::pow(row - centerRow, 2) + std::pow(col - centerCol, 2);

			// Scan bigger and bigger semicircles
			for(int r = dR, i = 0; r < rows + cols; r += 2 * dR, i = !i)
			{
				int R = r * r;
				if(dist < R && i)
				{
					px = Scale(row, 0, rows, realMin, realMax);
					break;
				}
				else if(dist < R && !i)
				{
					px = Scale(row, 0, rows, realMax, realMin);
					break;
				}
			}
		});

		return img;
	}

	cv::Mat Peaks(int rows, int cols, float minVal, float maxVal)
	{
		assert(rows > 0 && cols > 0 && "[Peaks] Invalid dimensions");


		// Swap values so the natural order is preserved
		if(minVal > maxVal)
		{
			std::swap(minVal, maxVal);
		}

		cv::Mat img{ rows,cols,CV_32FC1, cv::Scalar(0) };

		img.forEach<float>([&](float& px, const int* pos) -> void {
			float y = pu::Scale(pos[0], 0, rows - 1, -2.5, 2.5);
			float x = pu::Scale(pos[1], 0, cols - 1, -2.5, 2.5);
			px = Peak(x, y);
		});

		// Scale image so its values run from minVal to maxVal
		cv::normalize(img, img, minVal, maxVal, cv::NORM_MINMAX);

		return img;
	}

	void AddSaltPepperNoise(cv::Mat & img, float probability)
	{
		double minVal, maxVal;
		cv::minMaxIdx(img, &minVal, &maxVal);

		// Dont use opencv's forEach since its mulithreaded and 
		// will not generate same results every launch
		std::for_each(img.begin<float>(), img.end<float>(), [&](float& px) -> void {
			if(Noise() < probability)
			{
				px = Noise() > 0.5 ? minVal : maxVal;
			}
		});
	}

	void AddRandomNoise(cv::Mat & img, float probability, float magnitude)
	{
		double minVal, maxVal;
		cv::minMaxIdx(img, &minVal, &maxVal);

		// Dont use opencv's forEach since its mulithreaded and 
		// will not generate same results every launch
		std::for_each(img.begin<float>(), img.end<float>(), [&](float& px) -> void {
			if(Noise() < probability)
			{
				px = Scale(Noise(), 0, 1, minVal, maxVal);
			}
		});
	}
}


