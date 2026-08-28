#pragma once

#include <utility>

#include <include/list/ListBase.hpp>
#include <include/iterator/BasicIterator.hpp>

using namespace std;


namespace ndpp
{


/*
    ListIterator: The iterator which is used for List.
*/
class ListIterator : public ndpp_iterator::BasicIterator<NodeDevice>
{
public:
    inline ListIterator(pointer ptr) : ndpp_iterator::BasicIterator<NodeDevice>(ptr) {}

    inline ListIterator& operator++() override
    {
        this->_t_ptr = this->_t_ptr->_next;
        return *this;
    }

    inline ListIterator& operator--() override
    {
        this->_t_ptr = this->_t_ptr->_prev;
        return *this;
    }
};


/*
    List: It is a container that supports inserting generic data type, and it's implemented as a doubly-linked list.
    P.S Reference: https://cppreference.com/cpp/container/list
*/
class List : private ndpp_list_base::ListDevice
{
public:
    List() = default;
    inline List(List &&src);
    inline List(const List &src);

    virtual ~List() = default;

    /* 
        Used for the specified situation.
        List dst;
        dst = func(...);
    */
    inline List& operator=(const List &src);

    // Returns the number of elements.
    inline const size_t size() const;

    // Returns an iterator to the beginning.
    inline ListIterator begin() const;
    
    // Returns an iterator to the end.
    inline ListIterator end() const;

    // Returns a reverse iterator to the beginning.
    inline ListIterator rbegin() const;

    // Returns a reverse iterator to the end
    inline ListIterator rend() const;

    // Returns a reference to the first element in the container.
    inline NodeDevice& front() const;

    // Returns a reference to the last element in the container.
    inline NodeDevice& back() const;

    // Adds an element to the end (used for move constructor).
    inline void pushBack(Generic &&object);

    // Adds an element to the end (used for copy constructor).
    inline void pushBack(const Generic &object);

    // Inserts an element to the beginning (used for move constructor).
    inline void pushFront(Generic &&object);

    // Inserts an element to the beginning (used for copy constructor).
    inline void pushFront(const Generic &object);

    // Insert the node in front/rear of specified location (used for move constructor).
    // P.S Time complexity: O(1).
    inline void insert(ListIterator position, Generic &&object, const ndpp_action::ListPushAction action);

    // Insert the node in front/rear of specified location (used for copy constructor).
    // P.S Time complexity: O(1).
    inline void insert(ListIterator position, const Generic &object, const ndpp_action::ListPushAction action);

    // Erases element via specified location.
    inline void erase(ListIterator position);

    // Removes the last element.
    inline void popBack();

    // Removes the first element.
    inline void popFront();
    
    // Clears the contents.
    inline void clear();

    // Copy another List to itself.
    inline void copy(const List &src);

    // Clone itself to new List.
    inline List clone() const;

    // List migration.
    inline void migrate(List &src);

    // Moves elements from another List.
    void splice(ListIterator position, List &src);

    // Reverses the order of the elements in the container.
    void reverse();
};

inline List::List(List &&src) : ndpp_list_base::ListDevice(std::move(src))
{
}

inline List::List(const List &src) : ndpp_list_base::ListDevice(src)
{
}

inline List& List::operator=(const List &src)
{
    if (this != &src)
    {
        copy(src);
    }

    return *this;
}

inline const size_t List::size() const
{
    return NumNodes();
}

inline ListIterator List::begin() const 
{ 
    return ListIterator(this->_head); 
}
    
inline ListIterator List::end() const 
{ 
    return ListIterator(nullptr); 
}

inline ListIterator List::rbegin() const
{
    return ListIterator(this->_tail); 
}

inline ListIterator List::rend() const
{
    return ListIterator(nullptr);
}

inline NodeDevice& List::front() const
{
    if (!this->_head)
    {
        ndpp_log::logger("List.hpp", "List::front()", ndpp_log::RuntimeType::Error, 
                         "The list is empty.", true);
        exit(EXIT_FAILURE);
    }
    return *this->_head;
}

inline NodeDevice& List::back() const
{
    if (!this->_tail)
    {
        ndpp_log::logger("List.hpp", "List::back()", ndpp_log::RuntimeType::Error, 
                         "The list is empty.", true);
        exit(EXIT_FAILURE);
    }
    return *this->_tail;
}

inline void List::pushBack(Generic &&object)
{
    PushNodeToTail(CreateNode(std::move(object)));
}

inline void List::pushBack(const Generic &object)
{
    PushNodeToTail(CreateNode(object));
}

inline void List::pushFront(Generic &&object)
{
    PushNodeToHead(CreateNode(std::move(object)));
}

inline void List::pushFront(const Generic &object)
{
    PushNodeToHead(CreateNode(object));
}

inline void List::insert(ListIterator position, Generic &&object, const ndpp_action::ListPushAction action)
{
    InsertNode(&(*position), CreateNode(std::move(object)), action);
}

inline void List::insert(ListIterator position, const Generic &object, const ndpp_action::ListPushAction action)
{
    InsertNode(&(*position), CreateNode(object), action);
}

inline void List::erase(ListIterator position)
{
    EraseNode(&(*position));
}

inline void List::popBack()
{
    EraseNode(this->_tail);
}
    
inline void List::popFront()
{
    EraseNode(this->_head);
}

inline void List::clear()
{
    ClearNodes();
}

inline void List::copy(const List &src)
{
    CopyNodes(src);
}

inline List List::clone() const
{
    List _dst;
    _dst.copy(*this);
    return _dst;
}

inline void List::migrate(List &src)
{
    DeviceMigrate(src);
}


}; // namespace ndpp