import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  // PLO Trainer question banks are read with fs at runtime; make sure they ship
  // with the trainer's serverless functions.
  outputFileTracingIncludes: {
    "/api/trainer/**": ["./data/trainer/**"],
  },
};

export default nextConfig;
