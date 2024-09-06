import OpenAI from "https://deno.land/x/openai@v4.57.3/mod.ts";

const emojiData = await Deno.readTextFile("./emoji_data.json");
let emojiDataAsObjects = JSON.parse(emojiData);
emojiDataAsObjects = emojiDataAsObjects.filter((emoji: any) => {
  if (emoji.skintone_combination !== "") {
    return emoji.hexcode === emoji.skintone_base_hexcode;
  }
  return true;
});
const emojiDataAsStings = emojiDataAsObjects.map((emoji: any) =>
  JSON.stringify(emoji)
);

const client = new OpenAI();

console.log("creating embedding for data");

let index = 0;
for (const emojiData of emojiDataAsObjects) {
  const emojiDataString = emojiDataAsStings[index];
  const dataEmbedding = await client.embeddings.create({
    input: emojiDataString,
    model: "text-embedding-3-small",
  });
  emojiData.embedding = dataEmbedding.data[0].embedding;
  console.log(index);
  console.log(dataEmbedding.usage);
  console.log(emojiData.emoji);
  index++;
  if (index > 1000) {
    break;
  }
}

await Deno.writeTextFile(
  "./emoji_data_with_embeddings.json",
  JSON.stringify(emojiDataAsObjects.slice(0, 1000))
);
