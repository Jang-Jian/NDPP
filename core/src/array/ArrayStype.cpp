#include <include/array/ArrayStype.hpp>


namespace ndpp
{

bool same(const ndpp::SizeTArray &src1, const ndpp::SizeTArray &src2)
{
    size_t src1_size = src1.size();
    size_t src2_size = src2.size();

    if (src1_size != src2_size)
    {
        return false;
    }

    if (src1_size == 1)
    {
        return (src1[0] != src2[0]) ? false : true;
    }
    else if (src1_size == 2)
    {
        #pragma GCC unroll 2
        for (size_t index = 0; index < 2; ++index) 
        {
            if (src1[index] != src2[index])
            {
                return false;
            }
        }
    }
    else if (src1_size == 3)
    {
        #pragma GCC unroll 3
        for (size_t index = 0; index < 3; ++index) 
        {
            if (src1[index] != src2[index])
            {
                return false;
            }
        }
    }
    else if (src1_size == 4)
    {
        #pragma GCC unroll 4
        for (size_t index = 0; index < 4; ++index) 
        {
            if (src1[index] != src2[index])
            {
                return false;
            }
        }
    }
    else if (src1_size == 5)
    {
        #pragma GCC unroll 5
        for (size_t index = 0; index < 5; ++index) 
        {
            if (src1[index] != src2[index])
            {
                return false;
            }
        }
    }
    else if (src1_size == 6)
    {
        #pragma GCC unroll 6
        for (size_t index = 0; index < 6; ++index) 
        {
            if (src1[index] != src2[index])
            {
                return false;
            }
        }
    }
    else
    {
        for (size_t index = 0; index < 6; ++index) 
        {
            if (src1[index] != src2[index])
            {
                return false;
            }
        }
    }

    return true;
}

}; // namespace ndpp