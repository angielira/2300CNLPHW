/** Static export: `npm run build` writes a plain website to the `out` folder,
 *  which any static host (Netlify, Vercel, GitHub Pages) can serve. */
const nextConfig = {
  output: "export",
  images: { unoptimized: true },
  trailingSlash: true,
}
export default nextConfig
