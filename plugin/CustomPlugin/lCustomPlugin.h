#ifndef TRT_L_Custom_PLUGIN_H
#define TRT_L_Custom_PLUGIN_H
#include "NvInferPlugin.h"
#include "kernel.h"
#include "plugin.h"
#include <cassert>
#include <iostream>
#include <string>
#include <vector>

namespace nvinfer1
{
namespace plugin
{


// 为了后续方便注册，自定义插件Custom类需要继承nvinfer1::IPluginV2DynamicExt接口
class Custom : public nvinfer1::IPluginV2DynamicExt
{
public:
    // 用于在clone阶段，复制这个plugin时会用到的构造函数
    Custom(float negSlope);

    // 用于在deserialize阶段
    Custom(const void* buffer, size_t length);

    // 注意需要把默认构造函数删掉
    Custom() = delete;

    ~Custom() override = default;

    int getNbOutputs() const noexcept override;

    DimsExprs getOutputDimensions(int outputIndex, const nvinfer1::DimsExprs* inputs, 
                int nbInputs, nvinfer1::IExprBuilder& exprBuilder) noexcept override;

    int initialize() noexcept override;

    // 析构函数则需要执行terminate，terminate函数就是释放这个op之前开辟的一些显存空间
    void terminate() noexcept override;

    size_t getWorkspaceSize(const nvinfer1::PluginTensorDesc* inputs,
            int nbInputs, const nvinfer1::PluginTensorDesc* outputs, int nbOutputs) const noexcept override;

    int enqueue(const nvinfer1::PluginTensorDesc* inputDesc, const nvinfer1::PluginTensorDesc* outputDesc,
        const void* const* inputs, void* const* outputs, void* workspace, cudaStream_t stream) noexcept override;

    size_t getSerializationSize() const noexcept override;

    void serialize(void* buffer) const noexcept override;

    bool supportsFormatCombination(int pos, const nvinfer1::PluginTensorDesc* inOut, 
        int nbInputs, int nbOutputs) noexcept override;
    const char* getPluginType() const noexcept override;

    const char* getPluginVersion() const noexcept override;

    void destroy() noexcept override;

    nvinfer1::IPluginV2DynamicExt* clone() const noexcept override;

    // 另外需要增加一些重写方法
    void setPluginNamespace(const char* pluginNamespace) noexcept override;

    const char* getPluginNamespace() const noexcept override;

    nvinfer1::DataType getOutputDataType(int index, const nvinfer1::DataType* inputTypes, int nbInputs) const noexcept override;

    void attachToContext(cudnnContext* cudnn, cublasContext* cublas, nvinfer1::IGpuAllocator* allocator) noexcept override;
    void detachFromContext() noexcept override;

    void configurePlugin(const nvinfer1::DynamicPluginTensorDesc* in, int nbInputs, 
                       const nvinfer1::DynamicPluginTensorDesc* out, int nbOutputs) noexcept override;

private:
    float mNegSlope;
    int mBatchDim;

    // 增加一些属性
    std::string mPluginNamespace;
    std::string mNamespace;
};



// 另外需要写一个创建Custom插件工厂类
class CustomPluginCreator : public BaseCreator
{
public:
    CustomPluginCreator();

    ~CustomPluginCreator() override = default;

    const char* getPluginName() const noexcept override;

    const char* getPluginVersion() const noexcept override;

    const PluginFieldCollection* getFieldNames() noexcept override;

    IPluginV2DynamicExt* createPlugin(const char* name, const PluginFieldCollection* fc) noexcept override;

    IPluginV2DynamicExt* deserializePlugin(const char* name, const void* serialData, size_t serialLength) noexcept override;

private:
    static PluginFieldCollection mFC;
    static std::vector<PluginField> mPluginAttributes;
};

} // namespace plugin
} // namespace nvinfer1

#endif // TRT_L_RELU_PLUGIN_H
