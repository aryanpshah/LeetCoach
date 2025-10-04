import ProblemCard from "../components/ProblemCard";
import { PROBLEMS } from "../lib/problems";

const extraMeta = {
  "two-sum": {
    desc: "Find two numbers in an array that add to a target. Aim for O(n) time.",
    skills: ["Hash map reasoning", "Edge-case spotting", "Brute force vs optimal"]
  },
  "longest-substring": {
    desc: "Find the length of the longest substring without repeating characters.",
    skills: ["Sliding window patterns", "Set operations", "Time complexity narration"]
  },
  "merge-k-lists": {
    desc: "Merge k sorted linked lists into one sorted list efficiently.",
    skills: ["Heap design", "Pointer hygiene", "Complexity tradeoffs"]
  }
};

const problemCards = Object.values(PROBLEMS).map((problem) => {
  const meta = extraMeta[problem.id] ?? {};
  const cleanTitle = problem.title.replace(/^\d+\.\s*/, "");
  return {
    id: problem.id,
    title: cleanTitle,
    difficulty: problem.difficulty,
    tags: problem.tags,
    skills: meta.skills ?? [],
    desc: meta.desc ?? problem.statement.split("\n")[0].trim(),
    timeLimit: `${problem.timeLimitMin} minutes`
  };
});

export default function ProblemSelect() {
  return (
    <main className="container py-5 problem-select">
      <header className="text-center mb-5">
        <h1 className="display-5 fw-bold text-white mb-3">Choose Your Challenge</h1>
        <p className="lead text-secondary mb-0">
          Select a problem to begin your mock interview session.
        </p>
      </header>

      <div className="row row-cols-1 row-cols-md-3 g-4">
        {problemCards.map((problem) => (
          <ProblemCard key={problem.id} {...problem} />
        ))}
      </div>
    </main>
  );
}
