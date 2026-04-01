"""
MicroBlog API
Handles CRUD operations for micro blog posts, replies, reactions, and topics
"""
from flask import Blueprint, request, jsonify, g
from flask_restful import Api, Resource
from api.authorize import token_required
from model.microblog import MicroBlog, Topic
from __init__ import db


microblog_api = Blueprint('microblog_api', __name__, url_prefix='/api')
api = Api(microblog_api)


class MicroBlogAPI:
  
   class _CRUD(Resource):
       """MicroBlog CRUD operations"""
      
       @token_required()
       def post(self):
           """Create a new micro blog post"""
           current_user = g.current_user
           body = request.get_json()
          
           if not body:
               return {'message': 'Request body is required'}, 400
          
           content = body.get('content')
           if not content:
               return {'message': 'Content is required'}, 400
          
           if len(content) > 280:
               return {'message': 'Content must be 280 characters or less'}, 400
          
           topic_id = body.get('topicId')
           topic_path = body.get('topicPath')
           data = body.get('data', {})
          
           if topic_path and not topic_id:
               try:
                   topic = Topic.get_or_create_for_page(
                       page_path=topic_path,
                       page_title=topic_path.replace('/', ' ').title(),
                       allow_anonymous=True
                   )
                   if topic:
                       topic_id = topic.id
                   else:
                       return {'message': f'Failed to create or find topic for path: {topic_path}'}, 500
               except Exception as topic_error:
                   return {'message': f'Error handling topic: {str(topic_error)}'}, 500
          
           try:
               microblog = MicroBlog(
                   user_id=current_user.id,
                   content=content,
                   topic_id=topic_id,
                   data=data
               )
              
               created_microblog = microblog.create()
               if not created_microblog:
                   return {'message': 'Failed to create micro blog post'}, 500
              
               return jsonify(created_microblog.read())
              
           except ValueError as e:
               return {'message': str(e)}, 400
           except Exception as e:
               return {'message': f'Error creating micro blog post: {str(e)}'}, 500
      
       # ── CHANGED: removed @token_required() so guests can read posts ──
       def get(self):
           """Get micro blog posts with optional filtering (public read)"""
           limit     = request.args.get('limit', 200, type=int)
           topic_id  = request.args.get('topicId', type=int)
           page_path = request.args.get('pagePath')
           user_id   = request.args.get('userId', type=int)
           search    = request.args.get('search')
          
           try:
               if search:
                   microblogs = MicroBlog.search_content(search, limit)
               elif topic_id:
                   microblogs = MicroBlog.get_by_topic(topic_id, limit)
               elif page_path:
                   topic = Topic.get_by_page_path(page_path)
                   if topic:
                       microblogs = MicroBlog.get_by_topic(topic.id, limit)
                   else:
                       microblogs = []
               elif user_id:
                   microblogs = MicroBlog.get_by_user(user_id, limit)
               else:
                   microblogs = MicroBlog.get_all(limit)
               return jsonify({
                   'microblogs': microblogs,
                   'count': len(microblogs)
               })
           except Exception as e:
               return {'message': f'Error retrieving micro blog posts: {str(e)}'}, 500
      
       @token_required()
       def put(self):
           """Update a micro blog post"""
           current_user = g.current_user
           body = request.get_json()
          
           if not body:
               return {'message': 'Request body is required'}, 400
          
           microblog_id = body.get('id')
           if not microblog_id:
               return {'message': 'MicroBlog ID is required'}, 400
          
           microblog = MicroBlog.get_by_id(microblog_id)
           if not microblog:
               return {'message': 'MicroBlog post not found'}, 404
          
           if microblog._user_id != current_user.id and getattr(current_user, 'role', None) != 'Admin':
               return {'message': 'Permission denied'}, 403
          
           try:
               content = body.get('content')
               data    = body.get('data')
               updated_microblog = microblog.update(content=content, data=data)
               return jsonify(updated_microblog.read())
           except ValueError as e:
               return {'message': str(e)}, 400
           except Exception as e:
               return {'message': f'Error updating micro blog post: {str(e)}'}, 500
      
       @token_required()
       def delete(self):
           """Delete a micro blog post"""
           current_user = g.current_user
           body = request.get_json()
          
           if not body:
               return {'message': 'Request body is required'}, 400
          
           microblog_id = body.get('id')
           if not microblog_id:
               return {'message': 'MicroBlog ID is required'}, 400
          
           microblog = MicroBlog.get_by_id(microblog_id)
           if not microblog:
               return {'message': 'MicroBlog post not found'}, 404
          
           if microblog._user_id != current_user.id and getattr(current_user, 'role', None) != 'Admin':
               return {'message': 'Permission denied'}, 403
          
           try:
               microblog.delete()
               return {'message': 'MicroBlog post deleted successfully'}, 200
           except Exception as e:
               return {'message': f'Error deleting micro blog post: {str(e)}'}, 500
  
   class _Reply(Resource):
       """Handle replies to micro blog posts"""
      
       @token_required()
       def post(self):
           """Add a reply to a micro blog post"""
           current_user = g.current_user
           body = request.get_json()
          
           if not body:
               return {'message': 'Request body is required'}, 400
          
           microblog_id  = body.get('postId') or body.get('microblogId')
           reply_content = body.get('content')
          
           if not microblog_id:
               return {'message': 'postId (or microblogId) is required'}, 400
          
           if not reply_content:
               return {'message': 'Reply content is required'}, 400
          
           microblog = MicroBlog.get_by_id(microblog_id)
           if not microblog:
               return {'message': 'MicroBlog post not found'}, 404
          
           try:
               reply = microblog.add_reply(current_user.id, reply_content)
               return jsonify({
                   'message': 'Reply added successfully',
                   'reply': reply,
                   'microblog': microblog.read()
               })
           except ValueError as e:
               return {'message': str(e)}, 400
           except Exception as e:
               return {'message': f'Error adding reply: {str(e)}'}, 500

       # ── Already public (no decorator) — no change needed ──
       def get(self):
           """Fetch replies for a specific microblog post (public)"""
           post_id = request.args.get('postId', type=int) or request.args.get('microblogId', type=int)
           if not post_id:
               return {'message': 'postId query param is required'}, 400
           microblog = MicroBlog.get_by_id(post_id)
           if not microblog:
               return {'message': 'MicroBlog post not found'}, 404
           replies = microblog.get_replies()
           return jsonify({'replies': replies, 'count': len(replies)})
       @token_required()

       def delete(self):
            """Delete a reply from a micro blog post"""
            current_user = g.current_user
            body = request.get_json()

            if not body:
                return {'message': 'Request body is required'}, 400

            microblog_id = body.get('postId') or body.get('microblogId')
            reply_id     = body.get('replyId')

            if not microblog_id:
                return {'message': 'postId is required'}, 400
            if not reply_id:
                return {'message': 'replyId is required'}, 400

            microblog = MicroBlog.get_by_id(microblog_id)
            if not microblog:
                return {'message': 'MicroBlog post not found'}, 404

            is_admin = getattr(current_user, '_role', None) == 'Admin'
            try:
                ok, msg = microblog.delete_reply(reply_id, current_user.id, is_admin)
                if not ok:
                    return {'message': msg}, 403
                return jsonify({'message': 'Reply deleted', 'microblog': microblog.read()})
            except Exception as e:
                return {'message': f'Error deleting reply: {str(e)}'}, 500
  
   class _Reaction(Resource):
       """Handle reactions to micro blog posts"""
       @token_required()
       def post(self):
           current_user = g.current_user
           body = request.get_json()

           if not body:
               return {'message': 'Request body is required'}, 400

           microblog_id  = body.get('microblogId') or body.get('postId')
           reaction_type = body.get('reactionType')

           if not microblog_id:
               return {'message': 'MicroBlog ID is required'}, 400
           if not reaction_type:
               return {'message': 'Reaction type is required'}, 400

           user_id = getattr(current_user, 'id', None)
           if not user_id:
               return {'message': 'Not authenticated'}, 401

           microblog = MicroBlog.get_by_id(microblog_id)
           if not microblog:
               return {'message': 'MicroBlog post not found'}, 404

           try:
               microblog.add_reaction(user_id, reaction_type)
               from __init__ import db
               db.session.refresh(microblog)
               return jsonify({'message': 'Reaction added successfully', 'microblog': microblog.read()})
           except Exception as e:
               return {'message': f'Error adding reaction: {str(e)}'}, 500
      
       @token_required()
       def delete(self):
           current_user = g.current_user
           body = request.get_json()
          
           if not body:
               return {'message': 'Request body is required'}, 400
          
           microblog_id  = body.get('microblogId') or body.get('postId')
           reaction_type = body.get('reactionType')
          
           if not microblog_id:
               return {'message': 'MicroBlog ID is required'}, 400
           if not reaction_type:
               return {'message': 'Reaction type is required'}, 400
          
           microblog = MicroBlog.get_by_id(microblog_id)
           if not microblog:
               return {'message': 'MicroBlog post not found'}, 404
          
           try:
               removed = microblog.remove_reaction(current_user.id, reaction_type)
               if removed:
                   return jsonify({'message': 'Reaction removed successfully', 'microblog': microblog.read()})
               else:
                   return {'message': 'Reaction not found'}, 404
           except Exception as e:
               return {'message': f'Error removing reaction: {str(e)}'}, 500


class TopicAPI:
  
   class _CRUD(Resource):
      
       @token_required()
       def post(self):
           current_user = g.current_user
          
           if getattr(current_user, 'role', None) != 'Admin':
               return {'message': 'Permission denied. Admin access required.'}, 403
          
           body = request.get_json()
           if not body:
               return {'message': 'Request body is required'}, 400
          
           page_path  = body.get('pagePath')
           page_title = body.get('pageTitle')
          
           if not page_path or not page_title:
               return {'message': 'Page path and title are required'}, 400
          
           existing_topic = Topic.get_by_page_path(page_path)
           if existing_topic:
               return {'message': 'Topic already exists for this page path'}, 400
          
           try:
               topic_data = {
                   'page_path':          page_path,
                   'page_title':         page_title,
                   'page_description':   body.get('pageDescription'),
                   'display_name':       body.get('displayName'),
                   'color':              body.get('color', '#007bff'),
                   'icon':               body.get('icon'),
                   'allow_anonymous':    body.get('allowAnonymous', False),
                   'moderated':          body.get('moderated', False),
                   'max_posts_per_user': body.get('maxPostsPerUser', 10),
                   'settings':           body.get('settings', {})
               }
               topic = Topic(**topic_data)
               created_topic = topic.create()
               if not created_topic:
                   return {'message': 'Failed to create topic'}, 500
               return jsonify(created_topic.read())
           except Exception as e:
               return {'message': f'Error creating topic: {str(e)}'}, 500
      
       def get(self):
           page_path   = request.args.get('pagePath')
           page_key    = request.args.get('pageKey')
           active_only = request.args.get('activeOnly', 'true').lower() == 'true'
           search      = request.args.get('search')
          
           try:
               if page_path:
                   topic = Topic.get_by_page_path(page_path)
                   if topic:
                       return jsonify(topic.read())
                   else:
                       return {'message': 'Topic not found for this page'}, 404
               elif page_key:
                   topic = Topic.get_by_page_key(page_key)
                   if topic:
                       return jsonify(topic.read())
                   else:
                       return {'message': 'Topic not found for this page key'}, 404
               elif search:
                   topics = Topic.search_by_title(search)
               else:
                   topics = Topic.get_all_active() if active_only else Topic.get_all()
               return jsonify({'topics': topics, 'count': len(topics)})
           except Exception as e:
               return {'message': f'Error retrieving topics: {str(e)}'}, 500
      
       @token_required()
       def put(self):
           current_user = g.current_user
           if getattr(current_user, 'role', None) != 'Admin':
               return {'message': 'Permission denied. Admin access required.'}, 403
          
           body = request.get_json()
           if not body:
               return {'message': 'Request body is required'}, 400
          
           topic_id = body.get('id')
           if not topic_id:
               return {'message': 'Topic ID is required'}, 400
          
           topic = Topic.query.get(topic_id)
           if not topic:
               return {'message': 'Topic not found'}, 404
          
           try:
               update_data  = {k: v for k, v in body.items() if k != 'id'}
               updated_topic = topic.update(**update_data)
               return jsonify(updated_topic.read())
           except Exception as e:
               return {'message': f'Error updating topic: {str(e)}'}, 500


   class _PageMicroblogs(Resource):
      
       def get(self, page_key):
           current_user = None
           try:
               from api.authorize import get_current_user
               current_user = get_current_user()
           except:
               pass
          
           limit = request.args.get('limit', 20, type=int)
          
           try:
               topic = Topic.get_by_page_key(page_key)
               if not topic:
                   return {'message': 'Page topic not found'}, 404
               if not topic._is_active:
                   return {'message': 'This discussion is currently disabled'}, 403
               if not topic._allow_anonymous and not current_user:
                   return {'message': 'Authentication required to view this discussion'}, 401
              
               user_id  = current_user.id if current_user else None
               posts    = topic.get_recent_posts(limit=limit, user_id=user_id)
               can_post = topic.can_user_post(current_user.id) if current_user else False
              
               return jsonify({
                   'topic':         topic.read(),
                   'microblogs':    posts,
                   'count':         len(posts),
                   'canPost':       can_post,
                   'userPostCount': topic.get_user_post_count(user_id) if user_id else 0
               })
           except Exception as e:
               return {'message': f'Error retrieving page microblogs: {str(e)}'}, 500


   class _AutoCreate(Resource):
      
       @token_required()
       def post(self):
           limit     = request.args.get('limit', 50, type=int)
           topic_id  = request.args.get('topicId', type=int)
           user_id   = request.args.get('userId', type=int)
           search    = request.args.get('search')
           page_path = request.args.get('pagePath')

           try:
               if search:
                   microblogs = MicroBlog.search_content(search, limit)
               elif topic_id:
                   microblogs = MicroBlog.get_by_topic(topic_id, limit)
               elif page_path:
                   topic = Topic.get_by_page_path(page_path)
                   if topic:
                       microblogs = MicroBlog.get_by_topic(topic.id, limit)
                   else:
                       return jsonify({'microblogs': [], 'count': 0, 'message': 'No topic found for this pagePath'}), 200
               elif user_id:
                   microblogs = MicroBlog.get_by_user(user_id, limit)
               else:
                   microblogs = MicroBlog.get_all(limit)

               return jsonify({'microblogs': microblogs, 'count': len(microblogs)})
           except Exception as e:
               return {'message': f'Error retrieving micro blog posts: {str(e)}'}, 500


# Register endpoints
api.add_resource(MicroBlogAPI._CRUD,     '/microblog',              endpoint='microblog_crud')
api.add_resource(MicroBlogAPI._Reply,    '/microblog/reply',        endpoint='microblog_reply')
api.add_resource(MicroBlogAPI._Reaction, '/microblog/reaction',     endpoint='microblog_reaction')

api.add_resource(TopicAPI._CRUD,          '/microblog/topics',                    endpoint='microblog_topic_crud')
api.add_resource(TopicAPI._PageMicroblogs,'/microblog/page/<string:page_key>',    endpoint='microblog_page_posts')
api.add_resource(TopicAPI._AutoCreate,    '/microblog/topics/auto-create',        endpoint='microblog_topic_autocreate')