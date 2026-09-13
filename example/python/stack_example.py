import ndpp


if __name__ == "__main__":
    test_stack = ndpp.stack()

    test_stack.push("jj")
    test_stack.push(123)
    test_stack.push({"test": 789})

    copy_tst_stack = test_stack.clone()
    test_stack.pop()

    print("All nodes in test_stack.")
    # show test_stack.
    while test_stack.size():
        print(test_stack.top())
        test_stack.pop()
    
    print("\nAll nodes in copy_tst_stack.")
    # show copy_tst_stack.
    while copy_tst_stack.size():
        print(copy_tst_stack.top())
        copy_tst_stack.pop()