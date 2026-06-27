#pragma once

#include <utility>
#include <typeinfo>

#include <include/logging/Logging.hpp>

using namespace std;


namespace ndpp
{

/*
    Generic: Generic data type implementation using type-erased.
    P.S The data is host-based.
*/
class Generic
{
private:
    // TypeInterface: Type interface.
    struct TypeInterface 
    {
        inline virtual ~TypeInterface() {}
        virtual TypeInterface* clone() const = 0;
        virtual const std::type_info& type() const = 0;
    };

    // TypeHolder: Holding a practical data type.
    template<typename T>
    struct TypeHolder : public TypeInterface 
    {
        T value;
        inline explicit TypeHolder(const T &v) : value(v) {}

        inline explicit TypeHolder(T &&v) : value(std::move(v)) {}

        inline TypeInterface* clone() const override
        {
            return new TypeHolder<T>(value);
        }

        inline const std::type_info& type() const override
        {
            return typeid(T);
        }
    };

    TypeInterface *_data = nullptr;

    // Used for detecting the error in Generic::value().
    template<typename T>
    inline void detect() const;

    // Destory the memory.
    inline void destory();

    // Get real data type name via std::type_info.
    // P.S Used for Linux & MacOS.
    // Reference: https://shengyu7697.github.io/cpp-typeid/
    const std::string typeInfo2Str(const std::type_info& info) const;

public:
    inline Generic();

    inline Generic(Generic &&data) noexcept;

    inline Generic(const Generic &data);

    template<typename T>
    inline Generic(T &&data);

    template<typename T>
    inline Generic(const T &data);

    inline virtual ~Generic();

    inline Generic& operator=(const Generic &data);

    template<typename T>
    inline void set(const T &data);

    template<typename T>
    inline T& value();

    template<typename T>
    inline const T& value() const;

    // Get self real data type name.
    inline const std::string name() const;
};

inline Generic::Generic() : _data(nullptr)
{
}

inline Generic::Generic(Generic &&data) noexcept : _data(data._data)
{
    data._data = nullptr;
}

inline Generic::Generic(const Generic &data) : _data(data._data ? data._data->clone() : nullptr)
{
}

template<typename T>
inline Generic::Generic(T &&data) : _data(new TypeHolder<typename std::decay<T>::type>(std::forward<T>(data)))
{
}

template<typename T>
inline Generic::Generic(const T &data)
{
    set<T>(data);
}

inline Generic::~Generic() 
{ 
    destory();
}

inline Generic& Generic::operator=(const Generic &data)
{
    if (this != &data)
    {
        destory();
        this->_data = data._data ? data._data->clone() : nullptr;
    }

    return *this;
}

template<typename T>
inline void Generic::set(const T &data)
{
    destory();
    this->_data = new TypeHolder<T>(data);
}

inline void Generic::destory()
{
    if (this->_data)
    {
        delete this->_data;
    }
    this->_data = nullptr;
}

template<typename T>
inline void Generic::detect() const
{
    if (!this->_data)
    {
        ndpp_log::logger("DataGeneric.hpp", "Generic::value<T>()", ndpp_log::RuntimeType::ERROR, 
                         "There is no data type which is allocated (data=nullptr).", true);
        exit(EXIT_FAILURE);
    }

    const std::type_info &assigned_dtype = typeid(T);
    const std::type_info &allocated_dtype = this->_data->type();

    if (assigned_dtype != allocated_dtype)
    {
        std::string assigned_dtype_str = typeInfo2Str(assigned_dtype);
        std::string allocated_dtype_str = typeInfo2Str(allocated_dtype);

        ndpp_log::logger("DataGeneric.hpp", "Generic::value<T>()", ndpp_log::RuntimeType::ERROR, 
                         "The assigned type is '" + assigned_dtype_str + "' but allocated type is '" + allocated_dtype_str + "'.", true);
        exit(EXIT_FAILURE);
    }
}

template<typename T>
inline T& Generic::value()
{
    detect<T>();
    return static_cast<TypeHolder<T>*>(this->_data)->value;
}

template<typename T>
inline const T& Generic::value() const
{
    detect<T>();
    return static_cast<const TypeHolder<T>*>(this->_data)->value;
}

inline const std::string Generic::name() const
{
    if (!this->_data)
    {
        ndpp_log::logger("DataGeneric.hpp", "Generic::name()", ndpp_log::RuntimeType::ERROR, 
                         "There is no data type which is allocated (data=nullptr).", true);
        exit(EXIT_FAILURE);
    }

    return typeInfo2Str(this->_data->type());
}

}; // namespace ndpp