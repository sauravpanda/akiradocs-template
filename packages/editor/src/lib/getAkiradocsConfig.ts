import { AkiraDocsConfig } from "@/types/AkiraConfigType";

const config = require("../../../akiradocs.config.json");

export function getAkiradocsConfig(): AkiraDocsConfig {
  // Cast the entire config to AkiraDocsConfig since we know it matches the shape
  return config as AkiraDocsConfig;
}
