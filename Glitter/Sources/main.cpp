// Local Headers
#include "glitter.hpp"
#include "mesh.hpp"
#include "shader.hpp"
#include "utility.hpp"
#include "vr.h"

// System Headers
#include <glad/glad.h>
#include <GLFW/glfw3.h>
#include <Eigen/Dense>
#include <ppl.h>
#include <ppltasks.h>
#include <glm/gtx/string_cast.hpp>

// Standard Headers
#include <cstdio>
#include <cstdlib>
#include <memory>
#include <algorithm>
#include <vector>
#include <queue>
#include <random>
#include <atomic>
#include <zmq.hpp>
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include <stb_image_write.h>
#include <filesystem>

using namespace Mirage;
using namespace Eigen;
using namespace concurrency;

typedef Array<float, Dynamic, Dynamic, RowMajor> Array_t;
typedef Array<unsigned char, Dynamic, Dynamic, RowMajor> Array_ub;
typedef Array<unsigned int, Dynamic, Dynamic, RowMajor> Array_ui;
typedef Array<std::complex<float>, Dynamic, Dynamic, RowMajor> Array_complex;
typedef std::tuple<float, int, int> queue_t; // (weight, index, LOD)
typedef std::tuple<size_t, size_t, size_t, size_t> edge_t; // (tri_id1, tri_idx1, tri_id2, tri_idx2) of an edge 

const float LOD2TESLEVEL[3] = { 1,4,32 };
const int NUMLOD = 2;
const int BUDGET = 5000;
const int LODCOST[2] = { 16,168 };

const float LAMBDA = 3.0f;
const float ECCOFFSET = 2.0f;

const float FOV = 110.0f;
const float PI = 3.1415926f;

const bool enableVR = true;

class application {
public:
	~application() {
		if constexpr (enableVR) vr.Shutdown();
	}

	void Init(GLFWwindow* window) {
		glPatchParameteri(GL_PATCH_VERTICES, 3);
		glEnable(GL_DEPTH_TEST);
		glPixelStorei(GL_PACK_ALIGNMENT, 1);

		// Setup OpenGL context
		mWindow = window;
		glfwWindowHint(GLFW_VISIBLE, GLFW_FALSE);
		glfwWindowHint(GLFW_CONTEXT_RELEASE_BEHAVIOR, GLFW_RELEASE_BEHAVIOR_NONE);
		mBGWindow = glfwCreateWindow(1, 1, "", NULL, mWindow);
		create_task([&] {
			glfwMakeContextCurrent(mBGWindow);
			glGenFramebuffers(1, &mBGFBO);
			glPixelStorei(GL_PACK_ALIGNMENT, 1);
			glfwMakeContextCurrent(NULL);
		});

		// Setup background
		mSkyShader.attach("background.vert").attach("background.frag").link();
		mSkybox = std::shared_ptr<Mesh>(new TexMesh("viking/viking-rotated.obj"));

		// Setup mesh
		mShader.attach("tessellation.vert").attach("tessellation.tcs").attach("tessellation.tes").attach("tessellation.frag").link().activate().bind("showGaze", 0);
		mObject = std::shared_ptr<Mesh>(new TessellationMesh("cman/cman.obj"));
		mShader.bind("displacementCof", 0.0f).bind("hasNormalMap", 0).bind("worldLightDir", glm::vec3({ 0.0, 1.0, 2.0 })).bind("diffuseOffset", 0.2f).bind("specularCoeff", 0.2f);

		std::dynamic_pointer_cast<TessellationMesh>(mObject)->mShader = &mShader;
		NUMFACES = mObject->getObjectNum();

		// Setup framebuffers
		if constexpr (enableVR) {
			if (!vr.Init()) printf("Failed to initialize VR\n");

			vr.setRenderTargetSize(&mWidth, &mHeight);
			mWidth = 1200;
			mHeight = 1344;

			InitFrameBuffer(m_leftFbo, m_leftFboTex, 3, mWidth * 2, mHeight * 2);
			InitFrameBuffer(m_rightFbo, m_rightFboTex, 3, mWidth * 2, mHeight * 2);
		}
		InitFrameBuffer(mFBO, mFBOTex, 3, mWidth * 2, mHeight * 2);
		InitFrameBuffer(mResizeFBO, mResizeFBOTex, 3, mWidth, mHeight);
		for (int i = 0; i < NUMLOD + 1; ++i) InitFrameBuffer(mPopFBO[i], mPopFBOTex[i], 3, mWidth, mHeight);

		// Setup members for popping calculation
		InitPoppingCaculation();

		// Setup variables for tessellation;
		mFaceLOD = std::vector<int>(NUMFACES, 0);
		mFaceTesLevel = std::vector<float>(3 * NUMFACES, LOD2TESLEVEL[0]);
		mEdgeTesLevel = std::vector<float>(3 * NUMFACES, LOD2TESLEVEL[0]);
		glGenBuffers(1, &mFaceTesBuffer);
		glBindBuffer(GL_ARRAY_BUFFER, mFaceTesBuffer);
		glBufferData(GL_ARRAY_BUFFER, sizeof(int) * NUMFACES * 3, mFaceTesLevel.data(), GL_DYNAMIC_DRAW);
		glGenBuffers(1, &mEdgeTesBuffer);
		glBindBuffer(GL_ARRAY_BUFFER, mEdgeTesBuffer);
		glBufferData(GL_ARRAY_BUFFER, sizeof(int) * NUMFACES * 3, mEdgeTesLevel.data(), GL_DYNAMIC_DRAW);
		glBindBuffer(GL_ARRAY_BUFFER, 0);
		std::dynamic_pointer_cast<TessellationMesh>(mObject->mSubMeshes[0])->SetupTesBuffer(mFaceTesBuffer, mEdgeTesBuffer);
		mEdgePair = CalculateEdgePair(mObject->mSubMeshes[0]->mIndices, mObject->mSubMeshes[0]->mVertices.size());
	}

	void Render() {
		if constexpr (enableVR) {
			mSystemTimestamp = glfwGetTime();
			auto viewMatrix = glm::lookAt(mEye, mLookAt, mUp);
			// Update pose and gaze
			auto oldGaze = mGaze;
			auto oldTimestamp = mTimestamp;
			bool updated = vr.getGaze(mGaze, mTimestamp);

			if (updated) {
				auto timeDiff = mTimestamp - oldTimestamp;
				auto gazeSpeed = acos(glm::dot(mGaze, oldGaze)) / PI * 180.0 / timeDiff;

				if (gazeSpeed > 180.0) mIsSaccade = true;
				else if (mIsSaccade) mIsSaccade = false;
			}
			glm::mat4 HMDPose;
			bool poseUpdated = vr.getMatrixPoseHead(HMDPose);
			if (poseUpdated) mViewMatrix = HMDPose * viewMatrix;

			glViewport(0, 0, mWidth * 2, mHeight * 2);
			// Render left eye
			mViewProjectionMatrix = vr.getProjectionMatrix(vr::Eye_Left) * vr.getMatrixPoseEye(vr::Eye_Left) * mViewMatrix;
			glBindFramebuffer(GL_FRAMEBUFFER, m_leftFbo);
			RenderImage();

			// Render right eye
			mViewProjectionMatrix = vr.getProjectionMatrix(vr::Eye_Right) * vr.getMatrixPoseEye(vr::Eye_Right) * mViewMatrix;
			glBindFramebuffer(GL_FRAMEBUFFER, m_rightFbo);
			RenderImage();

			// Submit to headset
			vr.submitFrame(m_leftFboTex[0], m_rightFboTex[0]);
			glBindFramebuffer(GL_FRAMEBUFFER, 0);
		}
		else {
			// Update gaze and VPmatrix
			mGaze = MouseToGaze();
			auto viewMatrix = glm::lookAt(mEye, mLookAt, mUp);
			auto projMatrix = glm::perspective(FOV / 180.0f * 3.14159f, (float)mWidth / mHeight, 0.1f, 50000.0f);
			mViewMatrix = viewMatrix;
			mViewProjectionMatrix = projMatrix * mViewMatrix;

			// Rendering
			glViewport(0, 0, mWidth * 2, mHeight * 2);
			glBindFramebuffer(GL_FRAMEBUFFER, mFBO); 
			RenderImage();
			glBindFramebuffer(GL_FRAMEBUFFER, 0);
		}

		// Copy render result to screen
		RenderToScreen();

		if (flag && !mBGWorking && !mBGUpdate) {
			CalculatePopping(mIsSaccade);
		} else if (mBGUpdate) {
			mFaceLOD = std::move(mNewLOD);
			// Update tessellation level
			mFaceTesLevel = std::move(mNewFaceTesLevel);
			glBindBuffer(GL_ARRAY_BUFFER, mFaceTesBuffer);
			glBufferSubData(GL_ARRAY_BUFFER, 0, sizeof(int) * NUMFACES * 3, mFaceTesLevel.data());
			mEdgeTesLevel = std::move(mNewEdgeTesLevel);
			glBindBuffer(GL_ARRAY_BUFFER, mEdgeTesBuffer);
			glBufferSubData(GL_ARRAY_BUFFER, 0, sizeof(int) * NUMFACES * 3, mEdgeTesLevel.data());
			mBGUpdate = false;
		}
	}

	void RenderImage() {
		glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
		glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

		mSkyShader.activate().bind("gaze", mGaze).bind("view", mViewMatrix).bind("viewProjection", mViewProjectionMatrix);
		mSkybox->draw(mSkyShader.get()); glCheckError();

		mShader.activate().bind("gaze", mGaze).bind("view", mViewMatrix).bind("viewProjection", mViewProjectionMatrix);
		mObject->draw(mShader.get()); glCheckError();
	}

	void RenderToScreen() {
		if constexpr (enableVR) glBindFramebuffer(GL_READ_FRAMEBUFFER, m_rightFbo);
		else glBindFramebuffer(GL_READ_FRAMEBUFFER, mFBO);
		glReadBuffer(GL_COLOR_ATTACHMENT0);
		glBindFramebuffer(GL_DRAW_FRAMEBUFFER, 0);
		glBlitFramebuffer(0, 0, mWidth * 2, mHeight * 2, 0, 0, windowWidth, windowHeight, GL_COLOR_BUFFER_BIT, GL_LINEAR);
	}

	void KeyCallBack(int key, int scancode, int action, int mods) {
		if (action == GLFW_PRESS) {
			switch (key)
			{
			case GLFW_KEY_Q:
				flag = true;
				break;
			default:
				break;
			}
		}
	}

	void CalculatePopping(bool isSaccade = false) {
		mBGWorking = true;
		auto t = glfwGetTime();

		// Render for each possible LOD
		for (int i = 1; i <= 1 + NUMLOD; ++i) {
			mObject->setUnifromLOD((float)LOD2TESLEVEL[i - 1]);
			glBindFramebuffer(GL_FRAMEBUFFER, mPopFBO[i - 1]);
			glViewport(0, 0, mWidth, mHeight);
			RenderImage();
		}
		mObject->setUnifromLOD(0);

		// Readback user's view
		if constexpr (enableVR) glBindFramebuffer(GL_READ_FRAMEBUFFER, m_rightFbo);
		else glBindFramebuffer(GL_READ_FRAMEBUFFER, mFBO);
		glBindFramebuffer(GL_DRAW_FRAMEBUFFER, mResizeFBO);
		glReadBuffer(GL_COLOR_ATTACHMENT1);
		glDrawBuffer(GL_COLOR_ATTACHMENT1);
		glBlitFramebuffer(0, 0, mWidth * 2, mHeight * 2, 0, 0, mWidth, mHeight, GL_COLOR_BUFFER_BIT, GL_NEAREST);
		glReadBuffer(GL_COLOR_ATTACHMENT2);
		glDrawBuffer(GL_COLOR_ATTACHMENT2);
		glBlitFramebuffer(0, 0, mWidth * 2, mHeight * 2, 0, 0, mWidth, mHeight, GL_COLOR_BUFFER_BIT, GL_NEAREST);

		glBindFramebuffer(GL_READ_FRAMEBUFFER, mResizeFBO);
		glReadBuffer(GL_COLOR_ATTACHMENT1);
		glReadPixels(0, 0, mWidth, mHeight, GL_RED, GL_UNSIGNED_BYTE, mCurrId0.data());
		glReadPixels(0, 0, mWidth, mHeight, GL_GREEN, GL_UNSIGNED_BYTE, mCurrId1.data());
		glReadBuffer(GL_COLOR_ATTACHMENT2);
		glReadPixels(0, 0, mWidth, mHeight, GL_RED, GL_FLOAT, mCurrEcc.data());
		glBindFramebuffer(GL_FRAMEBUFFER, 0);


		create_task([this, isSaccade, t] {
			glfwMakeContextCurrent(mBGWindow);
			// Readback for each possible LOD
			glBindFramebuffer(GL_FRAMEBUFFER, mBGFBO);
			for (int i = 0; i < 1 + NUMLOD; ++i) {
				glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, mPopFBOTex[i][1], 0);
				glReadBuffer(GL_COLOR_ATTACHMENT0);
				glReadPixels(0, 0, mWidth, mHeight, GL_BLUE, GL_UNSIGNED_BYTE, mImage[i].data());
			}
			glBindFramebuffer(GL_FRAMEBUFFER, 0);

			// Ecc importance for user's view
			mCurrId = mCurrId0.cast<unsigned int>() + mCurrId1.cast<unsigned int>() * 256;
			mTriEcc = 0;
			for (size_t i = 0; i < mHeight; ++i) {
				for (size_t j = 0; j < mWidth; ++j) {
					mTriEcc[mCurrId(i, j)] += mCurrEcc(i, j) + ECCOFFSET;
				}
			}
			mTriEcc[0] = 0;

			FFTCalculation(isSaccade);

			// Initialize heap
			std::vector<queue_t> container;
			for (int idx = 0; idx < NUMFACES; ++idx) {
				int lod = mFaceLOD[idx];
				if (lod < NUMLOD && mTriEcc[idx] != 0) container.emplace_back(GetWeight(idx, lod, isSaccade), idx, lod + 1);
			}
			if (container.empty()) {
				flag = false;
				mBGWorking = false;
				glfwMakeContextCurrent(NULL);
				return;
			}
			std::priority_queue<queue_t> heap(std::less<queue_t>(), std::move(container));
			mNewLOD = mFaceLOD;

			// Update LOD using heap
			int budget = BUDGET;
			while (!heap.empty()) {
				auto [w, idx, lod] = heap.top();
				heap.pop();
				if (budget < LODCOST[lod - 1])continue;
				budget -= LODCOST[lod - 1];
				mNewLOD[idx] = lod;
				if (lod < NUMLOD) heap.emplace(GetWeight(idx, lod, isSaccade), idx, lod + 1);
			}

			TessellationEdgeHandler();

			std::cout << glfwGetTime() - t << std::endl;
			mBGUpdate = true;
			mBGWorking = false;
			glfwMakeContextCurrent(NULL);
		});
	}

	float GetWeight(int idx, int lod, bool isSaccade) const
	{
		if (!isSaccade) return (mTriEcc[idx] - LAMBDA * mTriPopping[lod][idx]) / LODCOST[lod];
		else return mTriPopping[lod][idx] / LODCOST[lod];
	}

	void TessellationEdgeHandler()
	{
		// Handle edge consistency
		std::vector<int> newEdgeLOD(3 * NUMFACES);
		for (size_t i = 0; i < mNewLOD.size(); ++i) {
			newEdgeLOD[3 * i] = mNewLOD[i];
			newEdgeLOD[3 * i + 1] = mNewLOD[i];
			newEdgeLOD[3 * i + 2] = mNewLOD[i];
		}

		for (auto& edge : mEdgePair) {
			auto [t1, i1, t2, i2] = edge;
			auto maxLOD = std::max(mNewLOD[t1], mNewLOD[t2]);
			newEdgeLOD[3 * t1 + i1] = maxLOD;
			newEdgeLOD[3 * t2 + i2] = maxLOD;
		}

		// Map LOD to tessellation level
		mNewFaceTesLevel = std::vector<float>(3 * NUMFACES);
		mNewEdgeTesLevel = std::vector<float>(3 * NUMFACES);
		for (size_t i = 0; i < NUMFACES; ++i) {
			auto faceTesLevel = LOD2TESLEVEL[mNewLOD[i]];
			mNewFaceTesLevel[3 * i] = faceTesLevel;
			mNewFaceTesLevel[3 * i + 1] = faceTesLevel;
			mNewFaceTesLevel[3 * i + 2] = faceTesLevel;
			mNewEdgeTesLevel[3 * i] = LOD2TESLEVEL[newEdgeLOD[3 * i]];
			mNewEdgeTesLevel[3 * i + 1] = LOD2TESLEVEL[newEdgeLOD[3 * i + 1]];
			mNewEdgeTesLevel[3 * i + 2] = LOD2TESLEVEL[newEdgeLOD[3 * i + 2]];
		}
	}

	void InitPoppingCaculation() {
		// Pre-allocate memory
		mCurrId = Array_ui(mHeight, mWidth);
		mCurrId0 = Array_ub(mHeight, mWidth);
		mCurrId1 = Array_ub(mHeight, mWidth);
		mCurrEcc = Array_t(mHeight, mWidth);
		mTriEcc = ArrayXf(NUMFACES);
		mTriPopping = std::vector<ArrayXf>(NUMLOD, ArrayXf(NUMFACES));
		mTriPoppingNomask = std::vector<ArrayXf>(NUMLOD, ArrayXf(NUMFACES));
		InitFFT();
	}

	std::vector<edge_t> CalculateEdgePair(const std::vector<GLuint>& indices, size_t numVert) {
		auto hashPair = [](const std::pair<size_t, size_t>& p) { return p.first ^ p.second; };
		auto compPair = [](const std::pair<size_t, size_t>& p1, const std::pair<size_t, size_t>& p2) { return p1.first == p2.first && p1.second == p2.second; };
		std::unordered_map<std::pair<size_t, size_t>, std::vector<size_t>, decltype(hashPair), decltype(compPair)> edgeMap(16, hashPair, compPair); // Edge -> Triangle

		// Calculate all edge -> triangle
		std::vector<size_t> v(3);
		for (size_t i = 0; i < indices.size() / 3; ++i) {
			v[0] = indices[3 * i];
			v[1] = indices[3 * i + 1];
			v[2] = indices[3 * i + 2];

			std::sort(v.begin(), v.end());
			edgeMap[std::make_pair(v[0], v[1])].push_back(i);
			edgeMap[std::make_pair(v[0], v[2])].push_back(i);
			edgeMap[std::make_pair(v[1], v[2])].push_back(i);
		}

		// Match triangles pair sharing edge
		std::vector<edge_t> edgePair;
		for (auto it = edgeMap.begin(); it != edgeMap.end(); ++it) {
			auto& k = it->first;
			auto& v = it->second;
			if (v.size() <= 1) continue;

			auto [v1, v2] = k;
			size_t pos1, pos2, t1 = v[0], t2 = v[1];
			for (size_t i = 0; i < 3; ++i) {
				if (indices[3 * t1 + i] != v1 && indices[3 * t1 + i] != v2) {
					pos1 = i;
					break;
				}
			}
			for (size_t i = 0; i < 3; ++i) {
				if (indices[3 * t2 + i] != v1 && indices[3 * t2 + i] != v2) {
					pos2 = i;
					break;
				}
			}
			edgePair.emplace_back(t1, pos1, t2, pos2);
		}

		return edgePair;
	}

	void FFTCalculation(bool isSaccade) {
		// ID and FFT
		parallel_for(0, NUMLOD + 1, [&](int l) {
			mIn[l] = mImage[l].cast<float>();
			fftwf_execute(mPlan[l]);
		});

		// Bandpass
		parallel_for(0, (NUMLOD + 1) * (int)mFreq.size(), [&](int i) {
			auto lod = i / mFreq.size();
			auto freqIdx = i % mFreq.size();

			mRIn[i] = mOut[lod] * mAFFilter[freqIdx];
			fftwf_execute(mRPlan[i]);
			mContrastIm[i] = mROut[i] / (mWidth * mHeight);

			mRIn[i] = mOut[lod] * mLFFilter[freqIdx];
			fftwf_execute(mRPlan[i]);
			mContrastIm[i] = mContrastIm[i] / (mROut[i] / (mWidth * mHeight)).max(0.1f) * mFreqSensitivity[freqIdx];
			if (!isSaccade)mContrastIm[i] = (mCurrEcc < (mFreq[freqIdx] / FOV)).select(0, mContrastIm[i]);
		});

		// Popping
		parallel_for(0, NUMLOD, [&](int l) {
			mPopping[l] = 0;
			for (int i = 0; i < mFreq.size(); ++i) {
				auto& m1 = mContrastIm[i + l * mFreq.size()];
				auto& m2 = mContrastIm[i + (l + 1) * mFreq.size()];
				mPopping[l] += abs(m1 - m2) / (abs(m1) + 10);
			}
		});

		// Assign to triangle by ID
		parallel_for(0, NUMLOD, [&](int l) {
			auto& triPopping = mTriPopping[l];
			triPopping = 0;
			for (size_t i = 0; i < mHeight; ++i) {
				for (size_t j = 0; j < mWidth; ++j) {
					triPopping[mCurrId(i, j)] += mPopping[l](i, j);
				}
			}
			triPopping[0] = 0;
		});
	}

	Array_t GenerateAF(float F) {
		auto norX = (float)mWidth / mHeight;
		auto norY = (float)mHeight / mHeight;

		Array<float, 1, Dynamic> x = Array<float, 1, Dynamic>::LinSpaced(mWidth / 2 + 1, 0, norX).square();
		Array<float, Dynamic, 1> y(mHeight);
		y.head(mHeight / 2 + 1) = Array<float, 1, Dynamic>::LinSpaced(mHeight / 2 + 1, 0, norY);
		y.tail(mHeight / 2) = Array<float, 1, Dynamic>::LinSpaced(mHeight / 2 + 1, norY, 0).head(mHeight / 2);
		y = y.square();

		auto meshgrid = (x.replicate(mHeight, 1) + y.replicate(1, mWidth / 2 + 1)).sqrt();
		Array_t filter = exp(-(meshgrid - F).square() / (2 * 0.3 * F * 0.3 * F));
		return filter;
	}

	Array_t GenerateLF(float F) {
		auto norX = (float)mWidth / mHeight;
		auto norY = (float)mHeight / mHeight;

		Array<float, 1, Dynamic> x = Array<float, 1, Dynamic>::LinSpaced(mWidth / 2 + 1, 0, norX).square();
		Array<float, Dynamic, 1> y(mHeight);
		y.head(mHeight / 2 + 1) = Array<float, 1, Dynamic>::LinSpaced(mHeight / 2 + 1, 0, norY);
		y.tail(mHeight / 2) = Array<float, 1, Dynamic>::LinSpaced(mHeight / 2 + 1, norY, 0).head(mHeight / 2);
		y = y.square();

		auto meshgrid = (x.replicate(mHeight, 1) + y.replicate(1, mWidth / 2 + 1)).sqrt();
		Array_t filter = exp(-meshgrid.square() / (2 * 0.3 * F * 0.3 * F));
		return filter;
	}

	void InitFFT() {
		// Pre-compute frequency and sensitivity
		const float L = 100.0;
		const float X = 10.0;
		const float b = 0.3 * pow(1 + 100 / L, 0.15);
		const float c = 0.06;

		auto s = [=](ArrayXf u) -> ArrayXf {
			float a = 540.0 * pow(1 + 0.7 / L, -0.2);
			auto aArray = a / (1 + 12 / (X * (1 + u / 3)));
			return aArray * u * exp(-b * u) * sqrt(1 + c * exp(b * u));
		};

		mFreq = Array<float, 9, 1>({ 1,2,4,8,16,32,64,128,256 });
		mFreqSensitivity = s(mFreq / FOV);

		for (int i = 0; i < mFreq.size(); ++i) {
			mAFFilter.push_back(GenerateAF(mFreq[i] / (mHeight / 2)));
			mLFFilter.push_back(GenerateLF(mFreq[i] / (mHeight / 2)));
		}

		mImage = std::vector<Array_ub>(NUMLOD + 1, Array_ub(mHeight, mWidth));
		mPopping = std::vector<Array_t>(NUMLOD, Array_t(mHeight, mWidth));
		mPoppingNomask = std::vector<Array_t>(NUMLOD, Array_t(mHeight, mWidth));
		mIn = std::vector<Array_t>(NUMLOD + 1, Array_t(mHeight, mWidth));
		mOut = std::vector<Array_complex>(NUMLOD + 1, Array_complex(mHeight, mWidth / 2 + 1));
		mPlan = std::vector<fftwf_plan>(NUMLOD + 1);
		for (int i = 0; i < NUMLOD + 1; ++i) mPlan[i] = fftwf_plan_dft_r2c_2d(mHeight, mWidth, mIn[i].data(), fftwf_cast(mOut[i].data()), FFTW_MEASURE);
		mRIn = std::vector<Array_complex>((NUMLOD + 1) * mFreq.size(), Array_complex(mHeight, mWidth / 2 + 1));
		mROut = std::vector<Array_t>((NUMLOD + 1) * mFreq.size(), Array_t(mHeight, mWidth));
		mRPlan = std::vector<fftwf_plan>((NUMLOD + 1) * mFreq.size());
		for (int i = 0; i < (NUMLOD + 1) * mFreq.size(); ++i) mRPlan[i] = fftwf_plan_dft_c2r_2d(mHeight, mWidth, fftwf_cast(mRIn[i].data()), mROut[i].data(), FFTW_MEASURE);
		mContrastIm = std::vector<Array_t>((NUMLOD + 1) * mFreq.size(), Array_t(mHeight, mWidth));
		mContrastImNomask = std::vector<Array_t>((NUMLOD + 1) * mFreq.size(), Array_t(mHeight, mWidth));
	}

	// Helper functions
	bool InitFrameBuffer(GLuint& fbo, GLuint* fboTex, int numColorTex, int width, int height) {
		// Generate FBO and textures
		glGenFramebuffers(1, &fbo);
		glBindFramebuffer(GL_FRAMEBUFFER, fbo);
		GLenum* bufs = new GLenum[numColorTex];

		// Setup textures
		for (int i = 0; i < numColorTex - 1; i++) {
			glGenTextures(1, &fboTex[i]);
			glBindTexture(GL_TEXTURE_2D, fboTex[i]);
			glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, width, height, 0, GL_RGBA, GL_UNSIGNED_BYTE, NULL);
			glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0 + i, GL_TEXTURE_2D, fboTex[i], 0);
			glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
			glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
			bufs[i] = GL_COLOR_ATTACHMENT0 + i;
		}
		glGenTextures(1, &fboTex[numColorTex - 1]);
		glBindTexture(GL_TEXTURE_2D, fboTex[numColorTex - 1]);
		glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA32F, width, height, 0, GL_RGBA, GL_FLOAT, NULL);
		glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0 + numColorTex - 1, GL_TEXTURE_2D, fboTex[numColorTex - 1], 0);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
		bufs[numColorTex - 1] = GL_COLOR_ATTACHMENT0 + numColorTex - 1;

		glGenTextures(1, &fboTex[numColorTex]);
		glBindTexture(GL_TEXTURE_2D, fboTex[numColorTex]);
		glTexImage2D(GL_TEXTURE_2D, 0, GL_DEPTH_COMPONENT, width, height, 0, GL_DEPTH_COMPONENT, GL_FLOAT, NULL);
		glFramebufferTexture2D(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_TEXTURE_2D, fboTex[numColorTex], 0);

		// Setup draw buffers
		glDrawBuffers(numColorTex, bufs);

		// Check status
		GLenum Status = glCheckFramebufferStatus(GL_FRAMEBUFFER); glCheckError();
		if (Status != GL_FRAMEBUFFER_COMPLETE) printf("FB error, status: 0x%x\n", Status); return false;

		// Cleanup
		delete[] bufs;
		glBindFramebuffer(GL_FRAMEBUFFER, 0);

		return true;
	}

	glm::vec3 MouseToGaze() {
		double x, y;
		glfwGetCursorPos(mWindow, &x, &y);
		y = (mHeight - 1) - y;
		x = (2.0 * x - mWidth) / mHeight;
		y = 2.0 * y / mHeight - 1.0;

		glm::vec3 dir = { x * tan(FOV / 2.0 / 180.0 * PI), y * tan(FOV / 2.0 / 180.0 * PI), -1 };
		return glm::normalize(dir);
	}

	void ResetLOD() {
		mUniformLODLevel = 0.0f;
		mFaceLOD = std::vector<int>(NUMFACES, 0);
		mFaceTesLevel = std::vector<float>(3 * NUMFACES, LOD2TESLEVEL[0]);
		mEdgeTesLevel = std::vector<float>(3 * NUMFACES, LOD2TESLEVEL[0]);
		glBindBuffer(GL_ARRAY_BUFFER, mFaceTesBuffer); glCheckError();
		glBufferSubData(GL_ARRAY_BUFFER, 0, sizeof(int) * NUMFACES * 3, mFaceTesLevel.data());
		glBindBuffer(GL_ARRAY_BUFFER, mEdgeTesBuffer);
		glBufferSubData(GL_ARRAY_BUFFER, 0, sizeof(int) * NUMFACES * 3, mEdgeTesLevel.data());
	}

	// Members
	// Dimensions
	uint32_t mWidth = windowWidth;
	uint32_t mHeight = windowHeight;
	
	// Camera related
	glm::mat4 mViewProjectionMatrix;
	glm::mat4 mViewMatrix;

	glm::vec3 mEye = { -1.979,0,1.148 };
	glm::vec3 mLookAt = { -1.272,0,0.4347 };
	glm::vec3 mUp = {0,1,0};

	// Objects and shaders
	Shader mShader;
	std::shared_ptr<Mesh> mObject;
	Shader mSkyShader;
	std::shared_ptr<Mesh> mSkybox;
	size_t NUMFACES;

	// Framebuffers
	GLuint mFBO;
	GLuint mFBOTex[4];
	GLuint mPopFBO[NUMLOD + 1];
	GLuint mPopFBOTex[NUMLOD + 1][4];
	GLuint m_leftFbo;
	GLuint m_leftFboTex[4];
	GLuint m_rightFbo;
	GLuint m_rightFboTex[4];
	GLuint mBGFBO;
	GLuint mResizeFBO;
	GLuint mResizeFBOTex[4];

	// Popping calculation pre-allocated memory
	Array_ui mCurrId;
	Array_ub mCurrId0;
	Array_ub mCurrId1;
	Array_t mCurrEcc;
	ArrayXf mTriEcc;
	std::vector<ArrayXf> mTriPopping;
	std::vector<ArrayXf> mTriPoppingNomask;

	ArrayXf mFreq;
	ArrayXf mFreqSensitivity;
	std::vector<Array_t> mAFFilter;
	std::vector<Array_t> mLFFilter;
	std::vector<Array_t> mPopping;
	std::vector<Array_t> mPoppingNomask;
	std::vector<Array_ub> mImage;
	std::vector<Array_t> mIn;
	std::vector<Array_complex> mOut;
	std::vector<fftwf_plan> mPlan;
	std::vector<Array_complex> mRIn;
	std::vector<Array_t> mROut;
	std::vector<fftwf_plan> mRPlan;
	std::vector<Array_t> mContrastIm;
	std::vector<Array_t> mContrastImNomask;

	// Tessellation related
	float mUniformLODLevel = 0.0f;
	std::vector<int> mFaceLOD;
	std::vector<int> mNewLOD;

	std::vector<float> mFaceTesLevel;
	std::vector<float> mEdgeTesLevel;
	std::vector<float> mNewFaceTesLevel;
	std::vector<float> mNewEdgeTesLevel;
	GLuint mFaceTesBuffer;
	GLuint mEdgeTesBuffer;
	std::vector<edge_t> mEdgePair;

	// Multithread related
	bool flag = false;
	GLFWwindow* mWindow;
	GLFWwindow* mBGWindow;
	std::atomic<bool> mBGWorking = false;
	std::atomic<bool> mBGUpdate = false;

	// VR related
	VRApplication vr;
	glm::vec3 mGaze = glm::vec3(0, 0, -1.0);
	double mTimestamp = 0;
	double mSystemTimestamp = 0;
	double mSaccadeEndTime;
	bool mIsSaccade = false;
};

int main() {

	// Load GLFW and Create a Window
	glfwInit();
	glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
	glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 0);
	glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
	glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);
	glfwWindowHint(GLFW_RESIZABLE, GL_FALSE);
	auto mWindow = glfwCreateWindow(windowWidth, windowHeight, "OpenGL", nullptr, nullptr);

	// Check for Valid Context
	if (mWindow == nullptr) {
		fprintf(stderr, "Failed to Create OpenGL Context");
		return EXIT_FAILURE;
	}

	// Create Context and Load OpenGL Functions
	glfwMakeContextCurrent(mWindow);
	gladLoadGL();
	fprintf(stderr, "OpenGL %s\n", glGetString(GL_VERSION));

	application app;
	app.Init(mWindow);

	glfwSetWindowUserPointer(mWindow, &app);
	auto key_callback = [](GLFWwindow* window, int key, int scancode, int action, int mods) {
		static_cast<application*>(glfwGetWindowUserPointer(window))->KeyCallBack(key, scancode, action, mods);
	};
	glfwSetKeyCallback(mWindow, key_callback);

	// Rendering Loop
	while (glfwWindowShouldClose(mWindow) == false) {
		if (glfwGetKey(mWindow, GLFW_KEY_ESCAPE) == GLFW_PRESS)
			glfwSetWindowShouldClose(mWindow, true);

		app.Render();

		// Flip Buffers and Draw
		glfwSwapBuffers(mWindow);
		glfwPollEvents();
	}
	glfwTerminate();
	return EXIT_SUCCESS;
}
