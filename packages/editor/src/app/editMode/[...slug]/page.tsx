import { redirect } from "next/navigation";

export default function SlugPage({ params }: { params: { slug: string[] } }) {
  redirect("/editmode");
}
