export const PROBLEMS = {
  "two-sum": {
    id: "two-sum",
    title: "1. Two Sum",
    difficulty: "Easy",
    timeLimitMin: 10,
    tags: ["Array", "Hash Map"],
    statement: `
Given an array of integers nums and an integer target, return indices of the two numbers such that they add up to target.

You may assume that each input would have exactly one solution, and you may not use the same element twice.

You can return the answer in any order.

Example:
Input: nums = [2,7,11,15], target = 9
Output: [0,1]
Explanation: Because nums[0] + nums[1] == 9, we return [0, 1].

Constraints:
- 2 <= nums.length <= 10^4
- -10^9 <= nums[i] <= 10^9
- -10^9 <= target <= 10^9
- Only one valid answer exists.

Follow-up:
Can you come up with an algorithm that is less than O(n^2) time complexity?
    `,
    io: {
      input: "nums = [2,7,11,15], target = 9",
      output: "[0,1]"
    },
    hint: "Use a hash map to store values and indices while scanning once."
  },

  "longest-substring": {
    id: "longest-substring",
    title: "3. Longest Substring Without Repeating Characters",
    difficulty: "Medium",
    timeLimitMin: 15,
    tags: ["String", "Sliding Window"],
    statement: `
Given a string s, find the length of the longest substring without repeating characters.

Example:
Input: s = "abcabcbb"
Output: 3
Explanation: The answer is "abc", with the length of 3.

Follow-up:
Can you do it in O(n) using a sliding window and a set or map?
    `,
    io: { input: `s = "abcabcbb"`, output: "3" },
    hint: "Sliding window with last-seen positions."
  },

  "merge-k-lists": {
    id: "merge-k-lists",
    title: "23. Merge k Sorted Lists",
    difficulty: "Hard",
    timeLimitMin: 20,
    tags: ["Linked List", "Heap"],
    statement: `
You are given an array of k linked-lists lists, each linked-list is sorted in ascending order.

Merge all the linked-lists into one sorted linked-list and return it.

Example:
Input: lists = [[1,4,5],[1,3,4],[2,6]]
Output: [1,1,2,3,4,4,5,6]

Follow-up:
Analyze time complexity for a heap-based solution vs pairwise merges.
    `,
    io: { input: "lists = [[1,4,5],[1,3,4],[2,6]]", output: "[1,1,2,3,4,4,5,6]" },
    hint: "Min-heap of current heads, pop and push next."
  }
};
