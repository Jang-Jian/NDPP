import ndpp


if __name__ == "__main__":
    test_queue = ndpp.queue()

    test_queue.push("jj")
    test_queue.push(123)
    test_queue.push({"test": 789})

    copy_tst_queue = test_queue.clone()
    test_queue.pop()

    print("All nodes in test_queue.")
    # show test_queue.
    while test_queue.size():
        print(test_queue.front())
        test_queue.pop()
    
    print("\nAll nodes in copy_tst_queue.")
    # show copy_tst_queue.
    while copy_tst_queue.size():
        print(copy_tst_queue.front())
        copy_tst_queue.pop()