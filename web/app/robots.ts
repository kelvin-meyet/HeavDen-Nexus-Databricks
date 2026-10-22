import type { MetadataRoute } from "next";

// Served as /robots.txt. Crawlers may index the site, but not each patient's page or the API:
// until 31 Oct those pages query Databricks' SQL warehouse, which bills while it is on.
export default function robots(): MetadataRoute.Robots {
  return {
    rules: { userAgent: "*", allow: "/", disallow: ["/patients/", "/api/"] },
  };
}
