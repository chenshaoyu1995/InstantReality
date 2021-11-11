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

const float UPDATEINTERVAL = 0.2f;
const float LOD2TESLEVEL[5] = { 1,2,4,8,16 };
const int NUMLOD = 4;
const int BUDGET = 200;
const int LODCOST[4] = { 4,12,24,48 };

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

		// Setup background
		mSkyShader.attach("skybox.vert").attach("skybox.frag").link();
		mSkybox = std::shared_ptr<Mesh>(new TexMesh("terrain/skybox_top.obj"));
		mSkybox->mSubMeshes.push_back(std::shared_ptr<Mesh>(new TexMesh("terrain/skybox_right.obj"))->mSubMeshes[0]);
		mSkybox->mSubMeshes.push_back(std::shared_ptr<Mesh>(new TexMesh("terrain/skybox_front.obj"))->mSubMeshes[0]);
		mSkybox->mSubMeshes.push_back(std::shared_ptr<Mesh>(new TexMesh("terrain/skybox_left.obj"))->mSubMeshes[0]);
		mSkybox->mSubMeshes.push_back(std::shared_ptr<Mesh>(new TexMesh("terrain/skybox_back.obj"))->mSubMeshes[0]);

		// Setup mesh
		mShader.attach("tessellation.vert").attach("tessellation.tcs").attach("tessellation.tes").attach("tessellation.frag").link().activate().bind("showGaze", 0);
		mObject = std::shared_ptr<Mesh>(new TessellationMesh("terrain/plane.obj"));
		mShader.bind("displacementCof", 10.0f).bind("worldLightDir", glm::vec3({ 0.0, 1.0, -2.0 })).bind("diffuseOffset", 0.4f).bind("specularCoeff", 0.2f).bind("hasNormalMap", true).bind("hasEmiss", false).bind("hasSpec", false);

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

		socket.bind("tcp://*:5555");

		std::string filename = PROJECT_SOURCE_DIR "/Glitter/Assets/mask.dat";
		ReadMatrix(filename, mEccMask);
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
			if (mSystemTimestamp > mUpdateTimestamp + UPDATEINTERVAL) {
				mUpdateTimestamp += UPDATEINTERVAL;
				CalculatePopping(mIsSaccade);
			}
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
		const static auto targetEye = glm::vec3({ 0.05, 2.90, 7.47 });
		if (action == GLFW_PRESS) {
			switch (key)
			{
			case GLFW_KEY_Q:
				flag = true;
				mUpdateTimestamp = mSystemTimestamp + UPDATEINTERVAL;

				auto inv = glm::inverse(enableVR ? vr.getMatrixPoseEye(vr::Eye_Right) * mViewMatrix : mViewMatrix);
				auto eye = inv * glm::vec4{ 0,0,0,1 };
				auto diff = targetEye - glm::vec3(eye);
				mEye += diff; mLookAt += diff;

				break;
			default:
				break;
			}
		}
	}

	void CalculatePopping(bool isSaccade = false) {
		mBGWorking = true;
		auto t = glfwGetTime();

		// Readback user's view
		if constexpr (enableVR) glBindFramebuffer(GL_READ_FRAMEBUFFER, m_rightFbo);
		else glBindFramebuffer(GL_READ_FRAMEBUFFER, mFBO);
		glBindFramebuffer(GL_DRAW_FRAMEBUFFER, mResizeFBO);
		glReadBuffer(GL_COLOR_ATTACHMENT1);
		glDrawBuffer(GL_COLOR_ATTACHMENT1);
		glBlitFramebuffer(0, 0, mWidth * 2, mHeight * 2, 0, 0, mWidth, mHeight, GL_COLOR_BUFFER_BIT, GL_NEAREST);

		glBindFramebuffer(GL_READ_FRAMEBUFFER, mResizeFBO);
		glReadBuffer(GL_COLOR_ATTACHMENT1);
		glReadPixels(0, 0, mWidth, mHeight, GL_RED, GL_UNSIGNED_BYTE, mCurrId0.data());
		glReadPixels(0, 0, mWidth, mHeight, GL_GREEN, GL_UNSIGNED_BYTE, mCurrId1.data());
		glBindFramebuffer(GL_FRAMEBUFFER, 0);

		auto gaze = mGaze;
		auto inv = glm::inverse(enableVR ? vr.getMatrixPoseEye(vr::Eye_Right) * mViewMatrix : mViewMatrix);
		auto eye = inv * glm::vec4{ 0,0,0,1 };
		auto lookat = inv * glm::vec4{ 0,0,-1,0 };
		auto up = inv * glm::vec4{ 0,1,0,0 };

		create_task([this, isSaccade, gaze, eye, lookat, up, t] {
			// Ecc importance for user's view
			auto coord = GazeToMouse(gaze);
			auto currEcc = mEccMask.block(mHeight - (int)roundf(coord.y), mWidth - (int)roundf(coord.x) + 40, mHeight, mWidth);

			mCurrId = mCurrId0.cast<unsigned int>() + mCurrId1.cast<unsigned int>() * 256;
			mCount = 0;
			mTriEcc = 0;
			for (size_t i = 0; i < mHeight; ++i) {
				for (size_t j = 0; j < mWidth; ++j) {
					mTriEcc[mCurrId(i, j)] += currEcc(i, j) + ECCOFFSET;
					mCount[mCurrId(i, j)]++;
				}
			}
			mCount[0] = 0;
			mTriEcc[0] = 0;

			// read from NN
			json nnInput;
			nnInput["eye"] = { eye[0], eye[1],eye[2] };
			nnInput["lookat"] = { lookat[0], lookat[1],lookat[2] };
			nnInput["up"] = { up[0], up[1], up[2] };
			nnInput["gaze"] = { gaze[0],gaze[1],gaze[2] };
			nnInput["isSaccade"] = (int)isSaccade;

			socket.send(zmq::buffer(nnInput.dump()), zmq::send_flags::none);
			zmq::message_t request;
			socket.recv(request, zmq::recv_flags::none);
			json reply = json::parse(request.to_string());

			for (int l = 0; l < NUMLOD; ++l) {
				auto jsonTriPopping = reply[std::to_string(l)];
				std::copy(jsonTriPopping.begin(), jsonTriPopping.end(), mTriPopping[l].begin());
				mTriPopping[l] *= 100.0f * mCount.cast<float>();
			}


			// Initialize heap
			std::vector<queue_t> container;
			for (int idx = 0; idx < NUMFACES; ++idx) {
				int lod = mFaceLOD[idx];
				if (lod < NUMLOD && mTriEcc[idx] != 0) container.emplace_back(GetWeight(idx, lod, isSaccade), idx, lod + 1);
			}
			if (container.empty()) {
				mBGWorking = false;
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
		mTriEcc = ArrayXf(NUMFACES);
		mCount = ArrayXf(NUMFACES);
		mTriPopping = std::vector<ArrayXf>(NUMLOD, ArrayXf(NUMFACES));
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

	glm::vec2 GazeToMouse(glm::vec3 gaze) {
		gaze /= -gaze.z;
		gaze /= tan(FOV / 2.0 / 180.0 * PI);

		double y = mHeight * (1 + gaze.y) / 2;
		double x = (mHeight * gaze.x + mWidth) / 2;

		x = std::clamp(x, -40.0, (double)mWidth - 40.0);
		y = std::clamp(y, 0.0, (double)mHeight);

		return { x,y };
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

	glm::vec3 mEye = { 0,1.5,8 };
	glm::vec3 mLookAt = { 0,0,0 };
	glm::vec3 mUp = glm::cross({ 1,0,0 }, mLookAt - mEye);

	// Objects and shaders
	Shader mShader;
	std::shared_ptr<Mesh> mObject;
	Shader mSkyShader;
	std::shared_ptr<Mesh> mSkybox;
	size_t NUMFACES;

	// Framebuffers
	GLuint mFBO;
	GLuint mFBOTex[4];
	GLuint m_leftFbo;
	GLuint m_leftFboTex[4];
	GLuint m_rightFbo;
	GLuint m_rightFboTex[4];
	GLuint mResizeFBO;
	GLuint mResizeFBOTex[4];

	// Popping calculation pre-allocated memory
	Array_ui mCurrId;
	Array_ub mCurrId0;
	Array_ub mCurrId1;
	Array_t mEccMask;
	ArrayXf mCount;
	ArrayXf mTriEcc;
	std::vector<ArrayXf> mTriPopping;

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
	std::atomic<bool> mBGWorking = false;
	std::atomic<bool> mBGUpdate = false;

	// VR related
	VRApplication vr;
	glm::vec3 mGaze = glm::vec3(0, 0, -1.0);
	double mTimestamp = 0;
	double mSystemTimestamp = 0;
	double mUpdateTimestamp = 0;
	double mSaccadeEndTime;
	bool mIsSaccade = false;

	//Network
	zmq::context_t context{ 1 };
	zmq::socket_t socket{ context, zmq::socket_type::req };
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
