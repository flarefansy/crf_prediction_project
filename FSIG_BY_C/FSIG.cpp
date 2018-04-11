
#include <stdio.h>
#include <stdlib.h>

const int pcaMatrixDim = 192;

const int outWidth = 16;
const int outHeight = 12;

void clip3(int &x, int minValue, int maxValue)
{
  if ( x < minValue )
  {
    x = minValue;
  }
  else if ( x > maxValue)
  {
    x = maxValue;
  }
}

void bicubicScale(unsigned char* frame, unsigned char* outputFrame, int inputWidth, int inputHeight, int outWidth, int outHeight)
{
  const double xScale = double(inputWidth) / outWidth;
  const double yScale = double(inputHeight) / outHeight;

  for (int i = 0; i < outHeight; i++ )
  {
    for (int j = 0; j < outWidth; j++ )
    {
      const int x = int(xScale * j);
      const int y = int(yScale * i);

      const double dx = xScale * j - x;
      const double dy = yScale * i - y;

      double C[4] = { 0 };

      for (int k = 0; k < 4; k++ )
      {
        int z = y - 1 + k;
        clip3(z, 0, inputHeight - 1);

        int x0 = x - 1;
        clip3(x0, 0, inputWidth - 1);

        int x1 = x;
        clip3(x1, 0, inputWidth - 1);

        int x2 = x + 1;
        clip3(x2, 0, inputWidth - 1);

        int x3 = x + 2;
        clip3(x3, 0, inputWidth - 1);

        int a0 = frame[z * inputWidth + x1];
        int d0 = frame[z * inputWidth + x0] - a0;
        int d2 = frame[z * inputWidth + x2] - a0;
        int d3 = frame[z * inputWidth + x3] - a0;

        double a1 = -1.0 / 3 * d0 + d2 - 1.0 / 6 * d3;
        double a2 = 1.0 / 2 * d0 + 1.0 / 2 * d2;
        double a3 = -1.0 / 6 * d0 - 1.0 / 2 * d2 + 1.0 / 6 * d3;

        C[k] = a0 + a1 * dx + a2 * dx * dx + a3 * dx * dx * dx;
      }

      double a0 = C[1];
      double d0 = C[0] - C[1];
      double d2 = C[2] - C[1];
      double d3 = C[3] - C[1];

      double a1 = -1.0 / 3 * d0 + d2 - 1.0 / 6 * d3;
      double a2 = 1.0 / 2 * d0 + 1.0 / 2 * d2;
      double a3 = -1.0 / 6 * d0 - 1.0 / 2 * d2 + 1.0 / 6 * d3;

      int output = int(a0 + a1 * dy + a2 * dy * dy + a3 * dy * dy * dy + 0.5);
      clip3(output, 0, 255);

      outputFrame[i * outWidth + j] = output;
    }
  }
}

int main( int argc, char* argv[] )
{
  if ( argc != 5 )
  {
    printf("error!");
  }

  char* inputYUVName = argv[1];
  const int inputWidth = atoi(argv[2]);
  const int inputHeight = atoi(argv[3]);
  const int numOfFrames = atoi(argv[4]);

  const int imageSize = inputWidth * inputHeight;
 
  FILE* inputYUV = NULL;
  fopen_s(&inputYUV, inputYUVName, "rb");

  FILE* outputExcel = NULL;
  fopen_s(&outputExcel, "fsig.csv", "a+");

  unsigned char* frame = NULL;
  frame = new unsigned char[imageSize];

  unsigned char* outputframe = NULL;
  outputframe = new unsigned char[outWidth * outHeight];

  const int kd = 32;
  double* seq = new double[outWidth * outHeight * numOfFrames];
  double* seqPCA = new double[kd * numOfFrames];

  for (int i = 0; i < numOfFrames; i++ )
  {
    fread(frame, sizeof(unsigned char), imageSize, inputYUV);
    fseek(inputYUV, imageSize / 2, SEEK_CUR);
    bicubicScale(frame, outputframe, inputWidth, inputHeight, outWidth, outHeight);

    for (int j = 0; j < outHeight; j++ )
    {
      for (int k = 0; k < outWidth; k++ )
      {
        seq[i * outWidth * outHeight + j * outWidth + k] = outputframe[j * outWidth + k];
      }
    }
  }

  double* pcaMatrix = NULL;
  pcaMatrix = new double[pcaMatrixDim * pcaMatrixDim];

  FILE* fsigMatrix = NULL;
  fopen_s(&fsigMatrix, "FSIG.txt", "rb");

  double temp = 0.0;
  for (int i = 0; i < pcaMatrixDim; i++ )
  {
    for (int j = 0; j < pcaMatrixDim; j++ )
    {
      fscanf_s(fsigMatrix, "%lf", &temp);
      pcaMatrix[i * pcaMatrixDim + j] = temp;
    }
  }

  for (int i = 0; i < numOfFrames; i++ )
  {
    for (int j = 0; j < kd; j++ )
    {
      seqPCA[i * kd + j] = 0.0;
      for (int k = 0; k < outWidth * outHeight; k++ )
      {
        seqPCA[i * kd + j] += seq[i * outWidth * outHeight + k] * pcaMatrix[k * outWidth * outHeight + j];
      }
    }
  }

  fseek(outputExcel, 0, SEEK_END);
  int iOffset = ftell(outputExcel);

  for (int i = 0; i < numOfFrames; i++ )
  {
    for (int j = 0; j < kd; j++ )
    {
      fprintf(outputExcel, "%.3lf,", seqPCA[i * kd + j]);
    }
  }
  fprintf(outputExcel, "\n");

  delete frame;
  delete outputframe;
  delete seqPCA;

  fclose(fsigMatrix);
  fclose(inputYUV);
  fclose(outputExcel);

  return 0;
}