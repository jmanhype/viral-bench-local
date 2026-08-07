#!/usr/bin/env node
// ego-browser Instagram post scraper
// Usage: ego-browser nodejs scripts/ig-scrape.js <post_url>
// Outputs JSON to stdout via cliLog

const url = process.argv[2] || 'https://www.instagram.com/p/CjPZxQKsYJN/'

// Extract shortcode
const m = url.match(/\/(p|reel|tv)\/([A-Za-z0-9_-]+)/)
if (!m) {
  cliLog(JSON.stringify({ success: false, error: 'Could not extract shortcode from URL' }))
  process.exit(1)
}
const shortcode = m[2]

const task = await useOrCreateTaskSpace('ig-scraper-' + shortcode)

// Enable CDP network monitoring for video URLs
await cdp('Network.enable', {})

// Collect video URLs from network responses
const videoUrls = []
let networkListenerActive = true

// Navigate to the post
try {
  await openOrReuseTab(url, { wait: true, timeout: 30 })
} catch (e) {
  cliLog(JSON.stringify({ success: false, error: 'Navigation failed: ' + e.message }))
  await completeTaskSpace(task.id, { keep: false })
  process.exit(1)
}

// Wait for content to load
await wait(5)

// Check if page loaded properly (not login wall or error)
const pageInfo_data = await pageInfo()
if (pageInfo_data.dialog) {
  await cdp('Page.handleJavaScriptDialog', { accept: true })
  await wait(2)
}

const pageText = await js(String.raw`document.body.innerText.substring(0, 500)`)
if (pageText.includes("Sorry, this page isn't available")) {
  cliLog(JSON.stringify({ success: false, error: 'Post not found or removed' }))
  await completeTaskSpace(task.id, { keep: false })
  process.exit(1)
}

// Extract all metadata from DOM in one pass
const metadata = await js(String.raw`(() => {
  const result = {
    shortcode: '${shortcode}',
    url: window.location.href,
    username: null,
    caption: null,
    like_count: null,
    comment_count: null,
    view_count: null,
    timestamp: null,
    media_type: null,
    images: [],
    videos: [],
    is_video: false
  }
  
  // Username - first link in header area
  const headerLinks = document.querySelectorAll('header a[href^="/"], article a[href^="/"]')
  for (const l of headerLinks) {
    const href = l.getAttribute('href')
    if (href && href.match(/^\/[^/]+\/$/) && !href.includes('/p/') && !href.includes('/reel/')) {
      result.username = href.replace(/\//g, '')
      break
    }
  }
  
  // Caption - longest text span that's not UI chrome
  const spans = document.querySelectorAll('span')
  let bestCaption = ''
  for (const s of spans) {
    const text = (s.innerText || '').trim()
    if (text.length > bestCaption.length && 
        !text.includes('Cookie') && 
        !text.includes('Sign up') &&
        !text.includes('Log in') &&
        text.length < 5000) {
      bestCaption = text
    }
  }
  if (bestCaption.length > 20) result.caption = bestCaption
  
  // Like count
  const bodyText = document.body.innerText
  const likeMatch = bodyText.match(/(\d[\d,.]*)\s*(likes?|others?)/i)
  if (likeMatch) {
    result.like_count = parseInt(likeMatch[1].replace(/,/g, ''))
  }
  
  // Comment count  
  const commentMatch = bodyText.match(/(\d[\d,.]*)\s*comments?/i)
  if (commentMatch) {
    result.comment_count = parseInt(commentMatch[1].replace(/,/g, ''))
  }
  
  // View count (reels)
  const viewMatch = bodyText.match(/(\d[\d,.]*)\s*views?/i)
  if (viewMatch) {
    result.view_count = parseInt(viewMatch[1].replace(/,/g, ''))
  }
  
  // Timestamp
  const timeEl = document.querySelector('time')
  if (timeEl) {
    result.timestamp = timeEl.getAttribute('datetime') || timeEl.innerText
  }
  
  // Images from CDN
  const imgs = document.querySelectorAll('img')
  const seen = new Set()
  for (const img of imgs) {
    const src = img.src || ''
    if (src.includes('cdninstagram.com') && !src.includes('emoji') && !seen.has(src)) {
      seen.add(src)
      result.images.push(src.split('?')[0])
    }
  }
  
  // Check if video
  const videos = document.querySelectorAll('video')
  result.is_video = videos.length > 0
  result.media_type = result.is_video ? 'video' : (result.images.length > 1 ? 'carousel' : 'image')
  
  return result
})()`)

// Try to capture video URL via CDP
if (metadata.is_video) {
  // Use CDP to get network responses with video content
  try {
    const events = await drainEvents()
    for (const evt of events) {
      if (evt.method === 'Network.responseReceived') {
        const respUrl = evt.params?.response?.url || ''
        const mimeType = evt.params?.response?.mimeType || ''
        if ((respUrl.includes('cdninstagram.com') || respUrl.includes('fbcdn.net')) && 
            (mimeType.includes('video') || respUrl.includes('.mp4'))) {
          videoUrls.push(respUrl.split('?')[0])
        }
      }
    }
  } catch(e) {}
  
  // Also reload and intercept if no video URLs found yet
  if (videoUrls.length === 0) {
    // Re-navigate to capture network events
    await gotoAndWait(url, { timeout: 20, settle: 5 })
    await wait(3)
    
    try {
      const events2 = await drainEvents()
      for (const evt of events2) {
        if (evt.method === 'Network.responseReceived') {
          const respUrl = evt.params?.response?.url || ''
          const mimeType = evt.params?.response?.mimeType || ''
          if ((respUrl.includes('cdninstagram.com') || respUrl.includes('fbcdn.net')) && 
              (mimeType.includes('video') || respUrl.includes('.mp4'))) {
            videoUrls.push(respUrl.split('?')[0])
          }
        }
      }
    } catch(e) {}
  }
  
  metadata.videos = [...new Set(videoUrls)]
}

// Clean up
await completeTaskSpace(task.id, { keep: false })

// Output result
metadata.success = true
cliLog(JSON.stringify(metadata))
