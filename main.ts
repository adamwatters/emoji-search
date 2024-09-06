import OpenAI from "https://deno.land/x/openai@v4.57.3/mod.ts";

const emojiData = await Deno.readTextFile("./emoji_data_with_embeddings.json");
const emojiDataAsObjects = JSON.parse(emojiData);
const client = new OpenAI();

// console.log("embedding created");
// console.log(dataEmbedding);

// // 1. User Input
const userInput = prompt("Enter a string:");

if (!userInput) {
  console.error("No user input provided");
  Deno.exit(1);
}

console.log("creating embedding for input");
const inputEmbedding = await client.embeddings.create({
  input: userInput,
  model: "text-embedding-3-small",
});

console.log("embedding created");
console.log(inputEmbedding);

const result = [];
for (const emojiData of emojiDataAsObjects) {
  if (!emojiData.embedding) {
    break;
  }
  const similarity = cosineSimilarity(
    emojiData.embedding,
    inputEmbedding.data[0].embedding
  );
  if (similarity > 0.24) {
    result.push({ emoji: emojiData.emoji, similarity });
  }
}

const sortedResult = result.sort((a, b) => b.similarity - a.similarity);
console.log(sortedResult.slice(0, 50));

function cosineSimilarity(vectorA: number[], vectorB: number[]) {
  if (vectorA.length !== vectorB.length) {
    throw new Error("Vectors must be of equal length");
  }

  const dotProduct = vectorA.reduce(
    (sum, value, index) => sum + value * vectorB[index],
    0
  );
  const magnitudeA = Math.sqrt(
    vectorA.reduce((sum, value) => sum + value * value, 0)
  );
  const magnitudeB = Math.sqrt(
    vectorB.reduce((sum, value) => sum + value * value, 0)
  );

  if (magnitudeA === 0 || magnitudeB === 0) {
    throw new Error("Magnitude of a vector cannot be zero");
  }

  return dotProduct / (magnitudeA * magnitudeB);
}
