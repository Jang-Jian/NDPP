import ndpp


if __name__ == "__main__":
    test_queue = ndpp.queue()

    test_queue.push("jj")
    test_queue.push(123)
    test_queue.push({"test": 789})

    print(test_queue.front())
    test_queue.pop()
    print(test_queue.front())
    test_queue.pop()
    print(test_queue.front())
    test_queue.pop()