#include <boost/python.hpp>
#include <boost/python/numpy.hpp>

#include <include/ndpp/ndpp_python.hpp>


BOOST_PYTHON_MODULE(_ndpp_cpython)
{
    try 
    {
        using namespace boost::python;
        using namespace boost::python::numpy;

        Py_Initialize();
        boost::python::numpy::initialize();

        scope().attr("__doc__") = "NDPP: Python wrapper for 'NUMERICAL DATA PIPELINE & PROCESSING'.";
        std::string _built_tags = "Python, ";
    #ifdef CUDA 
        _built_tags += "CUDA";
    #endif
    #ifdef HALF 
        _built_tags += ", HALF";
    #endif
        scope().attr("__version__") = "Built tags (Linux): " + _built_tags + ".";


        // Creating module name.
        // P.S Naming rule: _cc_{name}.
        auto ndppCPythonModule = [](const std::string &name) 
        {
            return std::string("_cc_") + name;
        };

    #ifdef CUDA
        def(ndppCPythonModule(std::string("set_cuda_device")).c_str(), &ndpp::ndpp_python::setCudaDevice);
        def(ndppCPythonModule(std::string("get_cuda_devices")).c_str(), &ndpp::ndpp_python::getCudaDevices);
    #endif
        def(ndppCPythonModule(std::string("check_cuda")).c_str(), &ndpp::ndpp_python::checkCuda);
    

        // Used for ndpp::ndpp_memory::DeviceStatus.
        enum_<ndpp::ndpp_memory::DeviceStatus>(ndppCPythonModule(std::string("device_status")).c_str()); 
        scope().attr(ndppCPythonModule(std::string(ndpp::ndpp_memory::_AllocationStr)).c_str()) = ndpp::Allocation;
        scope().attr(ndppCPythonModule(std::string(ndpp::ndpp_memory::_ReferenceStr)).c_str())  = ndpp::Reference;


        // Used for ndpp::ndpp_memory::ScalarType.
        enum_<ndpp::ndpp_memory::ScalarType>(ndppCPythonModule(std::string("scalar_type")).c_str()); 
        scope().attr(ndppCPythonModule(std::string(ndpp::ndpp_memory::_UInt8Str)).c_str())  = ndpp::UInt8;
        scope().attr(ndppCPythonModule(std::string(ndpp::ndpp_memory::_UInt16Str)).c_str()) = ndpp::UInt16;
        scope().attr(ndppCPythonModule(std::string(ndpp::ndpp_memory::_UInt32Str)).c_str()) = ndpp::UInt32;
        scope().attr(ndppCPythonModule(std::string(ndpp::ndpp_memory::_UInt64Str)).c_str()) = ndpp::UInt64;
        scope().attr(ndppCPythonModule(std::string(ndpp::ndpp_memory::_Int8Str)).c_str())   = ndpp::Int8;
        scope().attr(ndppCPythonModule(std::string(ndpp::ndpp_memory::_Int16Str)).c_str())  = ndpp::Int16;
        scope().attr(ndppCPythonModule(std::string(ndpp::ndpp_memory::_Int32Str)).c_str())  = ndpp::Int32;
        scope().attr(ndppCPythonModule(std::string(ndpp::ndpp_memory::_Int64Str)).c_str())  = ndpp::Int64;
    #ifdef HALF
        scope().attr(ndppCPythonModule(std::string(ndpp::ndpp_memory::_Float16Str)).c_str()) = ndpp::Float16;
    #endif
        scope().attr(ndppCPythonModule(std::string(ndpp::ndpp_memory::_Float32Str)).c_str()) = ndpp::Float32;
        scope().attr(ndppCPythonModule(std::string(ndpp::ndpp_memory::_Float64Str)).c_str()) = ndpp::Float64;
        scope().attr(ndppCPythonModule(std::string(ndpp::ndpp_memory::_BoolStr)).c_str())    = ndpp::Bool;


        // Used for ndpp::ndpp_log::RuntimeType.
        enum_<ndpp::ndpp_log::RuntimeType>(ndppCPythonModule(std::string("runtime_type")).c_str()); 
        scope().attr(ndppCPythonModule(std::string(ndpp::ndpp_log::_InfoStr)).c_str())  = ndpp::Info;
        scope().attr(ndppCPythonModule(std::string(ndpp::ndpp_log::_ErrorStr)).c_str()) = ndpp::Error;
        scope().attr(ndppCPythonModule(std::string(ndpp::ndpp_log::_WarnStr)).c_str())  = ndpp::Warn;
        scope().attr(ndppCPythonModule(std::string(ndpp::ndpp_log::_DebugStr)).c_str()) = ndpp::Debug;
        scope().attr(ndppCPythonModule(std::string(ndpp::ndpp_log::_EndStr)).c_str())   = ndpp::End;
    
        def(ndppCPythonModule(std::string("logger")).c_str(), &ndpp::ndpp_log::logger);


        // Used for ndpp::ndpp_memory::DeviceType.
        enum_<ndpp::ndpp_memory::DeviceType>(ndppCPythonModule(std::string("device_type")).c_str());
        scope().attr(ndppCPythonModule(std::string(ndpp::ndpp_memory::_HostStr)).c_str())        = ndpp::Host;
    #ifdef CUDA
        scope().attr(ndppCPythonModule(std::string(ndpp::ndpp_memory::_CudaDeviceStr)).c_str())  = ndpp::CudaDevice;
        scope().attr(ndppCPythonModule(std::string(ndpp::ndpp_memory::_CudaPinnedStr)).c_str())  = ndpp::CudaPinned;
        scope().attr(ndppCPythonModule(std::string(ndpp::ndpp_memory::_CudaUnifiedStr)).c_str()) = ndpp::CudaUnified;
        scope().attr(ndppCPythonModule(std::string(ndpp::ndpp_memory::_CudaZeroCpyStr)).c_str()) = ndpp::CudaZeroCpy;
    #endif


        // ndpp::ndpp_python::PyScalar: A wrapped ndpp::Scalar for Python.
        class_<ndpp::ndpp_python::PyScalar, boost::noncopyable>(ndppCPythonModule(std::string("scalar")).c_str(), init<>())
            .def("__init__", boost::python::make_constructor(ndpp::ndpp_python::makePyScalar))
            .def(ndppCPythonModule(std::string("data")).c_str(), &ndpp::ndpp_python::PyScalar::pydata)
            .def(ndppCPythonModule(std::string("type")).c_str(), &ndpp::ndpp_python::PyScalar::type);


        // Used for ndpp::ndpp_python::PyOpPyTensor.
        enum_<ndpp::ndpp_python::PyOpPyTensor>(ndppCPythonModule(std::string("pyop_tensor")).c_str()); 
        scope().attr(ndppCPythonModule(std::string(ndpp::ndpp_python::_NoneStr)).c_str())     = ndpp::ndpp_python::None;
        scope().attr(ndppCPythonModule(std::string(ndpp::ndpp_python::_ReversedStr)).c_str()) = ndpp::ndpp_python::Reversed;
        
        // ndpp::ndpp_python::PyTensor allocation.
        def(ndppCPythonModule(std::string("zeros")).c_str(), &ndpp::ndpp_python::pyzeros);
        def(ndppCPythonModule(std::string("ones")).c_str(),  &ndpp::ndpp_python::pyones);
        def(ndppCPythonModule(std::string("full")).c_str(),  &ndpp::ndpp_python::pyfull);

        // ndpp::ndpp_python::PyTensor: A wrapped ndpp::Tensor for Python.
        class_<ndpp::ndpp_python::PyTensor>(ndppCPythonModule(std::string("tensor")).c_str(), init<>())
            .def(ndppCPythonModule(std::string("getitem")).c_str(), &ndpp::ndpp_python::getPyTensor)
            .def(ndppCPythonModule(std::string("setitem")).c_str(), &ndpp::ndpp_python::setPyTensor)
            .def("__int__",   &ndpp::ndpp_python::getValueFromPyTensor<ndpp::ndpp_memory::ScalarTypeToCppType<ndpp::ndpp_memory::ScalarType::Int64>::type>)
            .def("__float__", &ndpp::ndpp_python::getValueFromPyTensor<ndpp::ndpp_memory::ScalarTypeToCppType<ndpp::ndpp_memory::ScalarType::Float64>::type>)
            .def("__bool__",  &ndpp::ndpp_python::getValueFromPyTensor<ndpp::ndpp_memory::ScalarTypeToCppType<ndpp::ndpp_memory::ScalarType::Bool>::type>)
            .def(ndppCPythonModule(std::string("add_tensor")).c_str(),      &ndpp::ndpp_python::addPyTensor)
            .def(ndppCPythonModule(std::string("sub_tensor")).c_str(),      &ndpp::ndpp_python::subPyTensor)
            .def(ndppCPythonModule(std::string("mul_tensor")).c_str(),      &ndpp::ndpp_python::mulPyTensor)
            .def(ndppCPythonModule(std::string("div_tensor")).c_str(),      &ndpp::ndpp_python::divPyTensor)
            .def(ndppCPythonModule(std::string("floordiv_tensor")).c_str(), &ndpp::ndpp_python::floorDivPyTensor)
            .def(ndppCPythonModule(std::string("iadd_tensor")).c_str(),     &ndpp::ndpp_python::iAddPyTensor)
            .def(ndppCPythonModule(std::string("isub_tensor")).c_str(),     &ndpp::ndpp_python::iSubPyTensor)
            .def(ndppCPythonModule(std::string("imul_tensor")).c_str(),     &ndpp::ndpp_python::iMulPyTensor)
            .def(ndppCPythonModule(std::string("idiv_tensor")).c_str(),     &ndpp::ndpp_python::iDivPyTensor)
            .def(ndppCPythonModule(std::string("dim")).c_str(), &ndpp::ndpp_python::PyTensor::dim)
            .def(ndppCPythonModule(std::string("elements")).c_str(), &ndpp::ndpp_python::PyTensor::elements)
            .def(ndppCPythonModule(std::string("allocations")).c_str(), &ndpp::ndpp_python::PyTensor::allocations)
            .def(ndppCPythonModule(std::string("scalar")).c_str(), &ndpp::ndpp_python::PyTensor::scalar)
            .def(ndppCPythonModule(std::string("device")).c_str(), &ndpp::ndpp_python::PyTensor::device)
            .def(ndppCPythonModule(std::string("status")).c_str(), &ndpp::ndpp_python::PyTensor::status)
            .def(ndppCPythonModule(std::string("destory")).c_str(), &ndpp::ndpp_python::PyTensor::destory)
            .def(ndppCPythonModule(std::string("copy")).c_str(), &ndpp::ndpp_python::PyTensor::pycopy)
            .def(ndppCPythonModule(std::string("clone")).c_str(), &ndpp::ndpp_python::PyTensor::pyclone)
            .def(ndppCPythonModule(std::string("to")).c_str(), &ndpp::ndpp_python::PyTensor::pyto)
            .def(ndppCPythonModule(std::string("sizes")).c_str(), &ndpp::ndpp_python::PyTensor::pysizes)
            .def(ndppCPythonModule(std::string("strides")).c_str(), &ndpp::ndpp_python::PyTensor::pystrides)
            .def(ndppCPythonModule(std::string("migrate")).c_str(), &ndpp::ndpp_python::PyTensor::pymigrate)
            .def(ndppCPythonModule(std::string("from_tensor")).c_str(), &ndpp::ndpp_python::PyTensor::pyfromtensor)
            .def(ndppCPythonModule(std::string("from_numpy")).c_str(), &ndpp::ndpp_python::PyTensor::pyfromnumpy);


        // Used for ndpp::ndpp_action::ListPushAction.
        enum_<ndpp::ndpp_action::ListPushAction>(ndppCPythonModule(std::string("list_push_action")).c_str()); 
        scope().attr(ndppCPythonModule(std::string(ndpp::ndpp_action::_FrontStr)).c_str()) = ndpp::Front;
        scope().attr(ndppCPythonModule(std::string(ndpp::ndpp_action::_BackStr)).c_str())  = ndpp::Back;  

        // ndpp::ndpp_python::PyNodeDevice: A wrapped ndpp::NodeDevice for Python.
        // P.S  Due to no constructor for 'node', we have to wrap it without tag '_cc_'.
        class_<ndpp::ndpp_python::PyNodeDevice>("node", no_init)
            .def("exists", &ndpp::ndpp_python::PyNodeDevice::pyexists)
            .def("data", &ndpp::ndpp_python::PyNodeDevice::pydata)
            .def("prev_node", &ndpp::ndpp_python::PyNodeDevice::pyprevnode)
            .def("next_node", &ndpp::ndpp_python::PyNodeDevice::pynextnode);

        // ndpp::ndpp_python::PyListIterator: Used for PyList.
        // P.S  Due to no constructor for 'list_iterator', we have to wrap it without tag '_cc_'.
        class_<ndpp::ndpp_python::PyListIterator>("list_iterator", no_init)
            .def("__iter__", &ndpp::ndpp_python::PyListIterator::pyiter, return_internal_reference<>())
            .def("__next__", &ndpp::ndpp_python::PyListIterator::pynext);
   
        // ndpp::ndpp_python::PyList: A wrapped ndpp::List for Python.
        class_<ndpp::ndpp_python::PyList>(ndppCPythonModule(std::string("list")).c_str(), init<>())
            .def(ndppCPythonModule(std::string("iter")).c_str(), +[](ndpp::ndpp_python::PyList& self){ return self.pybegin(); })
            .def(ndppCPythonModule(std::string("size")).c_str(), &ndpp::ndpp_python::PyList::size)
            .def(ndppCPythonModule(std::string("clear")).c_str(), &ndpp::ndpp_python::PyList::clear)
            .def(ndppCPythonModule(std::string("reverse")).c_str(),  &ndpp::ndpp_python::PyList::reverse)
            .def(ndppCPythonModule(std::string("erase")).c_str(),  &ndpp::ndpp_python::PyList::pyerase)
            .def(ndppCPythonModule(std::string("clone")).c_str(),  &ndpp::ndpp_python::PyList::pyclone)
            .def(ndppCPythonModule(std::string("migrate")).c_str(),  &ndpp::ndpp_python::PyList::pymigrate)
            .def(ndppCPythonModule(std::string("pop_front")).c_str(), &ndpp::ndpp_python::PyList::popFront)
            .def(ndppCPythonModule(std::string("pop_back")).c_str(),  &ndpp::ndpp_python::PyList::popBack)
            .def(ndppCPythonModule(std::string("front")).c_str(), &ndpp::ndpp_python::PyList::pyfront)
            .def(ndppCPythonModule(std::string("back")).c_str(),  &ndpp::ndpp_python::PyList::pyback)
            .def(ndppCPythonModule(std::string("splice")).c_str(),  &ndpp::ndpp_python::PyList::pysplice)
            .def(ndppCPythonModule(std::string("insert")).c_str(),     &ndpp::ndpp_python::iInsert)
            .def(ndppCPythonModule(std::string("push_back")).c_str(),  &ndpp::ndpp_python::iPushBack)
            .def(ndppCPythonModule(std::string("push_front")).c_str(), &ndpp::ndpp_python::iPushFront);
    } 
    catch (const boost::python::error_already_set&)
    {
        PyErr_Print(); 
    }
}