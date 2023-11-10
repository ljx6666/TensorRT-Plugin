#include "lCustomPlugin.h"
#include "checkMacrosPlugin.h"
#include "kernel.h"

using namespace nvinfer1;
using nvinfer1::plugin::CustomPluginCreator;
using nvinfer1::plugin::Custom;

static const char* Custom_PLUGIN_VERSION{"1"};
static const char* Custom_PLUGIN_NAME{"Custom_TRT"};
PluginFieldCollection CustomPluginCreator::mFC{};
std::vector<PluginField> CustomPluginCreator::mPluginAttributes;

// LeakyReLU {{{
Custom::Custom(float negSlope)
    : mNegSlope(negSlope)
    , mBatchDim(1)
{
}

Custom::Custom(const void* buffer, size_t length)
{
    const char *d = reinterpret_cast<const char *>(buffer), *a = d;
    mNegSlope = read<float>(d);
    mBatchDim = read<int>(d);
    ASSERT(d == a + length);
}


// 插件op返回多少个Tensor，比如MyCustomPlugin这个操作只输出一个Tensor(也就是一个output)，所以直接return 1
int Custom::getNbOutputs() const noexcept
{
    return 1;
}


// 根据输入个数和输入维度，获得第outputIndex个输出的维度，动态shape NCHW，而输入输出的之间的关系是通过exprBuilder来确定的，相当于一个四则运算器，做shape infer。
// 这个batch维度在getOutputDimensions中是可以获取到的
// 在这个成员函数中根据输入维度推理出模型的输出维度
DimsExprs Custom::getOutputDimensions(int outputIndex, const nvinfer1::DimsExprs* inputs, 
                int nbInputs, nvinfer1::IExprBuilder& exprBuilder) noexcept
{
    nvinfer1::DimsExprs output(inputs[0]);
    return output;
}


// 推理函数，可以设置max workspace，避免显存移除，并且可以显存复用。如果在前一层使用cudaMalloc申请了显存，下一层无法使用。另外，权值一般不复用，所以权值不会放在workspace里面，并且会
// 使用cudaMalloc进行申请，对于enqueue函数，由于动态shape都是不确定的数值，需要输入输出的描述。
// 如果我们的操作需要一些分布在显存中的中间变量，可以通过传过来的指针参数workspace获取
// 静态shape 信息是可以提前拿到的，而动态只有在运行的时候才能获得的。
int Custom::enqueue(const nvinfer1::PluginTensorDesc* inputDesc,
    const nvinfer1::PluginTensorDesc* outputDesc, const void* const* inputs, void* const* outputs, void* workspace,
    cudaStream_t stream) noexcept
{
    const void* inputData = inputs[0];
    void* outputData = outputs[0];
    //在common/kernel.h文件里按照IReluInference实现CustomInference
    pluginStatus_t status = CustomInference(stream, mBatchDim, mNegSlope, inputData, outputData);
    return status;
}


// 返回序列化时需要写多少字节到buffer中
size_t Custom::getSerializationSize() const noexcept
{
    // mNegSlope, mBatchDim
    return sizeof(float) + sizeof(int);
}


// 序列化函数，将plugin的参数权重写入到buffer中
// 把需要用的数据按照顺序序列化到buffer里头。
void Custom::serialize(void* buffer) const noexcept
{
    char *d = reinterpret_cast<char *>(buffer), *a = d;
    write(d, mNegSlope);
    write(d, mBatchDim);
    ASSERT(d == a + getSerializationSize());
}


// 判断pos索引的输入/输出是否支持inOut[pos].format和inOut[pos].type指定的格式/数据类型
bool Custom::supportsFormatCombination(int pos, const nvinfer1::PluginTensorDesc* inOut, 
        int nbInputs, int nbOutputs) noexcept
{
    assert(0 <= pos && pos < 2);
    const auto *in = inOut;
    const auto *out = inOut + nbInputs;
    switch (pos) {
        case 0:
        return in[0].type == DataType::kFLOAT &&
                in[0].format == nvinfer1::TensorFormat::kLINEAR;
        case 1:
        return out[0].type == in[0].type &&
                out[0].format == nvinfer1::TensorFormat::kLINEAR;
    }
}


// 初始化函数，在这个插件准备开始run之前执行。
// 主要初始化一些提前开辟空间的参数，一般是一些cuda操作需要的参数(例如conv操作需要执行卷积操作，我们就需要提前开辟weight和bias的显存)，
// 假如我们的算子需要这些参数，则在这里需要提前开辟显存。
// 需要注意的是，如果插件算子需要开辟比较大的显存空间，不建议自己去申请显存空间，可以使用Tensorrt官方接口传过来的workspace指针来获取显存空间。
// 因为如果这个插件被一个网络调用了很多次，而这个插件op需要开辟很多显存空间，那么TensorRT在构建network的时候会根据这个插件被调用的次数开辟很多显存，很容易导致显存溢出。
int Custom::initialize() noexcept
{
    return 0;
}



// terminate函数释放initialize开辟的一些显存空间
void Custom::terminate() noexcept {}


// 获得plugin所需要的显存大小，最好不要在plugin enqueue中使用cudaMalloc申请显存
// 这个函数需要返回这个插件op需要中间显存变量的实际数据大小(bytesize)，通过TensorRT的接口去获取，是比较规范的方式。
// 我们需要在这里确定这个op需要多大的显存空间去运行，在实际运行的时候就可以直接使用TensorRT开辟好的空间而不是自己去申请显存空间。
size_t Custom::getWorkspaceSize(const nvinfer1::PluginTensorDesc* inputs,
            int nbInputs, const nvinfer1::PluginTensorDesc* outputs, int nbOutputs) const noexcept
{
    return 0;
}


// 获得plugin的类型，用于反序列化使用
const char* Custom::getPluginType() const noexcept
{
    return Custom_PLUGIN_NAME;
}


// 获得plugin的版本，用于反序列化使用
const char* Custom::getPluginVersion() const noexcept
{
    return Custom_PLUGIN_VERSION;
}


// 释放整个plugin占用的资源
void Custom::destroy() noexcept
{
    delete this;
}


// 将这个plugin对象克隆一份给TensorRT的builder、network或者engine。这个成员函数会调用第二个构造函数
// clone成员函数主要用于传递不变的权重和参数，将plugin复制n多份，从而可以被不同engine或者builder或者network使用。
IPluginV2DynamicExt* Custom::clone() const noexcept
{
    // 将要克隆的plugin的权重和参数传递给这个构造函数。
    auto* plugin = new Custom(mNegSlope);
    plugin->setPluginNamespace(mPluginNamespace.c_str());
    plugin->initialize();
    return plugin;
}

void Custom::setPluginNamespace(const char* pluginNamespace) noexcept
{
    mPluginNamespace = pluginNamespace;
}


// 为这个插件设置namespace名字，如果不设置则默认是""，需要注意的是同一个namespace下的plugin如果名字相同会冲突。
const char* Custom::getPluginNamespace() const noexcept
{
    return mPluginNamespace.c_str();
}


// 根据输入个数和输入类型，获得第index个输出的类型
nvinfer1::DataType Custom::getOutputDataType(
    int index, const nvinfer1::DataType* inputTypes, int nbInputs) const noexcept
{
    ASSERT(inputTypes && nbInputs > 0 && index == 0);
    return inputTypes[0];
}


// Attach the plugin object to an execution context and grant the plugin the access to some context resource.
// 如果这个插件使用到了一些其他东西，例如cublas handle，可以直接借助TensorRT内部提供的cublas handle
void Custom::attachToContext(
    cudnnContext* cudnnContext, cublasContext* cublasContext, IGpuAllocator* gpuAllocator) noexcept
{
}


// Detach the plugin object from its execution context.
void Custom::detachFromContext() noexcept {}


// 配置插件op，判断输入和输出类型数量是否正确
void Custom::configurePlugin(const nvinfer1::DynamicPluginTensorDesc* in, int nbInputs,
    const nvinfer1::DynamicPluginTensorDesc* out, int nbOutputs) noexcept
{
    // Not support dynamic shape in C dimension
    ASSERT(nbInputs == 1 && in[0].desc.dims.d[1] != -1);
}


// 创建一个空的mPluginAttributes初始化mFC。
CustomPluginCreator::CustomPluginCreator()
{
    mPluginAttributes.clear();
    mPluginAttributes.emplace_back(PluginField("negSlope", nullptr, PluginFieldType::kFLOAT32, 1));

    mFC.nbFields = mPluginAttributes.size();
    mFC.fields = mPluginAttributes.data();
}


// 获得plugin name和version，用来辨识creator
const char* CustomPluginCreator::getPluginName() const noexcept
{
    return Custom_PLUGIN_NAME;
}

const char* CustomPluginCreator::getPluginVersion() const noexcept
{
    return Custom_PLUGIN_VERSION;
}


// 通过PluginFieldCollection去创建plugin，将op需要的权值和参数一个一个取出来
// 这个是成员变量，也会作为getFieldNames成员函数的返回类型。PluginFieldCollection的主要作用是传递这个插件op所需要的权重和参数，
// 在实际的engine推理过程中并不使用，而在parse中会用到(例如caffe2trt、onnx2trt)。
const PluginFieldCollection* CustomPluginCreator::getFieldNames() noexcept
{
    return &mFC;
}


// 这里的createPlugin需要自己写创建内容，然后自己去调用，create函数可以用来封装接口并向外界提供库，不对外暴露plugin的设计，本质是通过传递结构体参数，
// 利用其调用构造函数
// 这个成员函数作用是通过PluginFieldCollection去创建plugin，将op需要的权重和参数一个一个取出来，然后调用上文提到的第一个构造函数
IPluginV2DynamicExt* CustomPluginCreator::createPlugin(const char* name, const PluginFieldCollection* fc) noexcept
{
    const PluginField* fields = fc->fields;
    ASSERT(fc->nbFields == 1);
    ASSERT(fields[0].type == PluginFieldType::kFLOAT32);
    float negSlope = *(static_cast<const float*>(fields[0].data));
    Custom* obj = new Custom{negSlope};

    obj->setPluginNamespace(mNamespace.c_str());
    obj->initialize();
    return obj;
}


// 反序列化，调用反序列化那个构造函数，生成plugin
// 这个函数会被onnx-tensorrt的一个叫做TRT_PluginV2的转换op调用，这个op会读取onnx模型的data数据将其反序列化到network中。
IPluginV2DynamicExt* CustomPluginCreator::deserializePlugin(const char* name, const void* serialData, size_t serialLength) noexcept
{
    // This object will be deleted when the network is destroyed, which will
    // call CustomPlugin::destroy()
    Custom* obj = new Custom{serialData, serialLength};
    obj->setPluginNamespace(mNamespace.c_str());
    obj->initialize();
    return obj;
}
